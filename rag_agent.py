import os
import json
import requests
from typing import List, Dict, Any, Optional
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain_community.callbacks import StreamlitCallbackHandler
import streamlit as st
import re
from bs4 import BeautifulSoup
import time
import pandas as pd

class FabricSearchTool(BaseTool):
    """Tool for searching fabric information"""
    name: str = "fabric_search"
    description: str = "Search for fabric materials and their properties"
    database: Any = None
    
    def __init__(self, database):
        super().__init__()
        self.database = database
    
    def _run(self, query: str) -> str:
        """Search for fabric materials"""
        try:
            results = self.database.search_materials(query)
            if results:
                response = f"Found {len(results)} materials matching '{query}':\n\n"
                for material in results[:5]:  # Limit to top 5 results
                    response += f"• {material['name']} ({material['category']})\n"
                    response += f"  {material['description']}\n\n"
                return response
            else:
                return f"No materials found matching '{query}'"
        except Exception as e:
            return f"Error searching materials: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class WebSearchTool(BaseTool):
    """Tool for searching the web for fabric information"""
    name: str = "web_search"
    description: str = "Search the web for fabric properties and sustainability data"
    
    def _run(self, query: str) -> str:
        """Simulate web search (in real implementation, would use SerpAPI or similar)"""
        search_results = {
            "organic cotton": "Organic cotton uses 91% less water than conventional cotton and produces 46% less CO2. It's grown without synthetic pesticides and fertilizers.",
            "recycled polyester": "Recycled polyester (rPET) reduces energy consumption by 75% and CO2 emissions by 71% compared to virgin polyester. It's made from post-consumer plastic bottles.",
            "hemp": "Hemp requires 50% less water than cotton and can grow without pesticides. It produces 2-3 times more fiber per acre than cotton.",
            "bamboo": "Bamboo is highly sustainable, growing up to 3 feet per day without pesticides. However, processing can be chemically intensive.",
            "lyocell": "Lyocell (Tencel) is made from wood pulp using a closed-loop process that recycles 99% of solvents. It's biodegradable and requires less water than cotton."
        }
        
        query_lower = query.lower()
        for key, value in search_results.items():
            if key in query_lower:
                return f"Search results for '{query}':\n\n{value}"
        
        return f"Limited information available for '{query}'. Consider searching for specific fabric properties."
    
    async def _arun(self, query: str) -> str:
        return self._run(query)

class FabricDataExtractor:
    """Extract and structure fabric data from various sources"""
    
    def __init__(self, database):
        self.database = database
        self.search_tool = FabricSearchTool(database)
        self.web_tool = WebSearchTool()
    
    def extract_material_properties(self, material_name: str) -> Dict[str, Any]:
        """Extract comprehensive properties for a material"""
        # First check database
        db_results = self.database.search_materials(material_name)
        
        # Then search web for additional information
        web_results = self.web_tool._run(material_name)
        
        # Combine and structure the data
        properties = {
            'name': material_name,
            'database_info': db_results[0] if db_results else None,
            'web_info': web_results,
            'extracted_metrics': self._extract_metrics_from_text(web_results)
        }
        
        return properties
    
    def _extract_metrics_from_text(self, text: str) -> Dict[str, float]:
        """Extract numerical metrics from text using regex patterns"""
        metrics = {}
        
        # Define patterns for different metrics
        patterns = {
            'water_consumption': r'(\d+(?:\.\d+)?)\s*(?:L|liters?|kg|tons?)\s*(?:per|of|water)',
            'ghg_emissions': r'(\d+(?:\.\d+)?)\s*(?:kg|tons?)\s*(?:CO2|CO2e|emissions?)',
            'energy_consumption': r'(\d+(?:\.\d+)?)\s*(?:kWh|energy)',
            'cost': r'\$(\d+(?:\.\d+)?)\s*(?:per|kg|lb)',
            'tensile_strength': r'(\d+(?:\.\d+)?)\s*(?:MPa|strength)',
            'moisture_regain': r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s*(?:moisture|regain)'
        }
        
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    metrics[metric] = float(matches[0])
                except ValueError:
                    continue
        
        return metrics
    
    def validate_and_store_data(self, material_name: str, properties: Dict[str, Any]) -> bool:
        """Validate extracted data and store in database"""
        try:
            # Check if material exists
            existing_materials = self.database.search_materials(material_name)
            
            if not existing_materials:
                # Add new material
                material_id = self.database.add_material(
                    name=material_name,
                    category="Extracted",
                    description=properties.get('web_info', '')
                )
            else:
                material_id = existing_materials[0]['id']
            
            # Add extracted metrics
            for metric_name, value in properties.get('extracted_metrics', {}).items():
                self.database.add_metric(
                    material_id=material_id,
                    metric_name=metric_name,
                    value=value,
                    source_explanation=f"Extracted from web search: {properties.get('web_info', '')[:100]}..."
                )
            
            return True
        except Exception as e:
            st.error(f"Error storing data: {str(e)}")
            return False

class FabricAgent:
    """Main agent for fabric data discovery and optimization"""
    
    def __init__(self, database, openai_api_key: Optional[str] = None):
        self.database = database
        self.extractor = FabricDataExtractor(database)
        
        # Initialize LLM
        if openai_api_key:
            self.llm = ChatOpenAI(
                temperature=0,
                openai_api_key=openai_api_key,
                model="gpt-3.5-turbo"
            )
        else:
            self.llm = None
        
        # Set up tools
        self.tools = [
            self.extractor.search_tool,
            self.extractor.web_tool
        ]
    
    def search_and_extract(self, query: str) -> Dict[str, Any]:
        """Search for fabric information and extract structured data"""
        try:
            # Search database
            db_results = self.database.search_materials(query)
            
            # Search web
            web_results = self.extractor.web_tool._run(query)
            
            # Extract properties
            properties = self.extractor.extract_material_properties(query)
            
            return {
                'query': query,
                'database_results': db_results,
                'web_results': web_results,
                'extracted_properties': properties,
                'success': True
            }
        except Exception as e:
            return {
                'query': query,
                'error': str(e),
                'success': False
            }
    
    def get_optimization_recommendations(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get optimization recommendations based on requirements"""
        recommendations = []
        
        # Analyze requirements
        if requirements.get('sustainability_priority', False):
            # Recommend sustainable materials
            sustainable_materials = self.database.search_materials("organic hemp bamboo lyocell")
            recommendations.append({
                'type': 'sustainable_materials',
                'materials': sustainable_materials[:3],
                'reasoning': 'These materials have lower environmental impact'
            })
        
        if requirements.get('cost_priority', False):
            # Recommend cost-effective materials
            cost_effective = self.database.search_materials("recycled polyester cotton")
            recommendations.append({
                'type': 'cost_effective_materials',
                'materials': cost_effective[:3],
                'reasoning': 'These materials offer good value for money'
            })
        
        if requirements.get('durability_priority', False):
            # Recommend durable materials
            durable_materials = self.database.search_materials("nylon polyester glass fiber")
            recommendations.append({
                'type': 'durable_materials',
                'materials': durable_materials[:3],
                'reasoning': 'These materials have high tensile strength and durability'
            })
        
        return recommendations
    
    def generate_blend_suggestions(self, preferences: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate blend suggestions based on user preferences"""
        suggestions = []
        
        # Get materials data
        materials_df = self.database.get_materials_dataframe()
        
        # Define blend templates based on preferences
        blend_templates = [
            {
                'name': 'Sustainable Comfort Blend',
                'description': 'High comfort with environmental focus',
                'materials': ['Organic Cotton', 'Lyocell (Tencel)', 'Hemp'],
                'target_ratios': [40, 35, 25],
                'best_for': 'Casual wear, activewear'
            },
            {
                'name': 'Durable Performance Blend',
                'description': 'High durability for performance wear',
                'materials': ['Recycled Polyester (rPET)', 'Nylon', 'Spandex'],
                'target_ratios': [60, 30, 10],
                'best_for': 'Athletic wear, outdoor gear'
            },
            {
                'name': 'Luxury Sustainable Blend',
                'description': 'Premium feel with sustainability',
                'materials': ['Organic Cotton', 'Silk', 'Cashmere'],
                'target_ratios': [50, 30, 20],
                'best_for': 'Premium clothing, formal wear'
            },
            {
                'name': 'Cost-Effective Blend',
                'description': 'Good performance at lower cost',
                'materials': ['Recycled Polyester (rPET)', 'Cotton', 'Bamboo'],
                'target_ratios': [50, 30, 20],
                'best_for': 'Everyday wear, mass production'
            }
        ]
        
        # Filter suggestions based on preferences
        for template in blend_templates:
            if preferences.get('budget') == 'low' and 'Cost-Effective' in template['name']:
                suggestions.append(template)
            elif preferences.get('sustainability') == 'high' and 'Sustainable' in template['name']:
                suggestions.append(template)
            elif preferences.get('durability') == 'high' and 'Durable' in template['name']:
                suggestions.append(template)
            elif preferences.get('luxury') == 'high' and 'Luxury' in template['name']:
                suggestions.append(template)
        
        # If no specific preferences, return all suggestions
        if not suggestions:
            suggestions = blend_templates
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def analyze_blend_performance(self, blend: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the performance of a proposed blend"""
        try:
            materials_df = self.database.get_materials_dataframe()
            
            # Calculate weighted averages for each metric
            performance = {
                'environmental_score': 0,
                'cost_score': 0,
                'durability_score': 0,
                'comfort_score': 0
            }
            
            total_weight = sum(blend['ratios'])
            
            for material, ratio in zip(blend['materials'], blend['ratios']):
                weight = ratio / total_weight
                
                # Find material in database
                material_data = materials_df[materials_df['material'] == material]
                if not material_data.empty:
                    row = material_data.iloc[0]
                    
                    # Environmental metrics (lower is better)
                    env_metrics = ['water_consumption', 'ghg_emissions', 'land_use']
                    env_score = sum(row.get(metric, 0) for metric in env_metrics) / len(env_metrics)
                    performance['environmental_score'] += env_score * weight
                    
                    # Cost metrics (lower is better)
                    cost_metrics = ['raw_material_cost']
                    cost_score = sum(row.get(metric, 0) for metric in cost_metrics) / len(cost_metrics)
                    performance['cost_score'] += cost_score * weight
                    
                    # Durability metrics (higher is better)
                    dur_metrics = ['tensile_strength']
                    dur_score = sum(row.get(metric, 0) for metric in dur_metrics) / len(dur_metrics)
                    performance['durability_score'] += dur_score * weight
                    
                    # Comfort metrics (higher is better)
                    comf_metrics = ['moisture_regain']
                    comf_score = sum(row.get(metric, 0) for metric in comf_metrics) / len(comf_metrics)
                    performance['comfort_score'] += comf_score * weight
            
            # Normalize scores
            for key in performance:
                if 'environmental' in key or 'cost' in key:
                    performance[key] = max(0, 10 - performance[key])  # Invert for display
                else:
                    performance[key] = min(10, performance[key])
            
            return {
                'blend': blend,
                'performance': performance,
                'overall_score': sum(performance.values()) / len(performance),
                'recommendations': self._generate_performance_recommendations(performance)
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'blend': blend
            }
    
    def _generate_performance_recommendations(self, performance: Dict[str, float]) -> List[str]:
        """Generate recommendations based on performance scores"""
        recommendations = []
        
        if performance['environmental_score'] < 5:
            recommendations.append("Consider adding more sustainable materials like hemp or organic cotton")
        
        if performance['cost_score'] < 5:
            recommendations.append("Consider using recycled materials to reduce costs")
        
        if performance['durability_score'] < 5:
            recommendations.append("Consider adding synthetic fibers like nylon or polyester for durability")
        
        if performance['comfort_score'] < 5:
            recommendations.append("Consider adding natural fibers like cotton or silk for comfort")
        
        if not recommendations:
            recommendations.append("This blend shows good balance across all metrics")
        
        return recommendations

    def get_latest_insights(self) -> Dict[str, Any]:
        """Get latest material discoveries and insights"""
        try:
            # Get materials data
            materials_df = self.database.get_materials_dataframe()
            
            # Find materials with high sustainability scores
            if 'sustainability_score' not in materials_df.columns:
                # Calculate sustainability scores if not present
                env_metrics = ['water_consumption', 'ghg_emissions', 'land_use']
                available_env_metrics = [m for m in env_metrics if m in materials_df.columns]
                if available_env_metrics:
                    env_scores = []
                    for metric in available_env_metrics:
                        min_val = materials_df[metric].min()
                        max_val = materials_df[metric].max()
                        if max_val > min_val:
                            normalized = 10 * (1 - (materials_df[metric] - min_val) / (max_val - min_val))
                            env_scores.append(normalized)
                        else:
                            env_scores.append(pd.Series(5, index=materials_df.index))
                    
                    if env_scores:
                        materials_df['sustainability_score'] = pd.concat(env_scores, axis=1).mean(axis=1)
            
            # Get top sustainable materials
            if 'sustainability_score' in materials_df.columns:
                top_sustainable = materials_df.nlargest(3, 'sustainability_score')
                insights = []
                
                for _, row in top_sustainable.iterrows():
                    insights.append({
                        'name': row['material'],
                        'innovation': f"High sustainability score ({row['sustainability_score']:.2f}/10) with excellent environmental performance",
                        'score': row['sustainability_score']
                    })
                
                return {
                    'success': True,
                    'materials': insights
                }
            else:
                # Fallback to basic insights
                return {
                    'success': True,
                    'materials': [
                        {
                            'name': 'Organic Hemp',
                            'innovation': 'Low water consumption and high durability make it an excellent sustainable choice',
                            'score': 8.5
                        },
                        {
                            'name': 'Lyocell (Tencel)',
                            'innovation': 'Closed-loop production process with minimal environmental impact',
                            'score': 8.2
                        },
                        {
                            'name': 'Recycled Polyester',
                            'innovation': 'Reduces plastic waste while maintaining performance properties',
                            'score': 7.8
                        }
                    ]
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'materials': []
            }

    def analyze_sustainability_trends(self) -> Dict[str, Any]:
        """Analyze sustainability trends in the material database"""
        try:
            # Get materials data
            materials_df = self.database.get_materials_dataframe()
            
            trends = []
            
            # Analyze environmental metrics trends
            env_metrics = ['water_consumption', 'ghg_emissions', 'land_use']
            available_env_metrics = [m for m in env_metrics if m in materials_df.columns]
            
            if available_env_metrics:
                # Calculate average environmental impact by material type
                material_types = materials_df['material'].str.lower()
                
                # Natural fibers trend
                natural_fibers = materials_df[material_types.str.contains('cotton|hemp|bamboo|silk|wool', na=False)]
                if not natural_fibers.empty:
                    avg_env_impact = natural_fibers[available_env_metrics].mean().mean()
                    trends.append({
                        'category': 'Natural Fibers',
                        'trend': f"Average environmental impact: {avg_env_impact:.2f} (lower is better)",
                        'direction': 'improving' if avg_env_impact < 5 else 'needs attention'
                    })
                
                # Synthetic fibers trend
                synthetic_fibers = materials_df[material_types.str.contains('polyester|nylon|acrylic', na=False)]
                if not synthetic_fibers.empty:
                    avg_env_impact = synthetic_fibers[available_env_metrics].mean().mean()
                    trends.append({
                        'category': 'Synthetic Fibers',
                        'trend': f"Average environmental impact: {avg_env_impact:.2f} (lower is better)",
                        'direction': 'improving' if avg_env_impact < 5 else 'needs attention'
                    })
                
                # Recycled materials trend
                recycled_materials = materials_df[material_types.str.contains('recycled|rpet', na=False)]
                if not recycled_materials.empty:
                    avg_env_impact = recycled_materials[available_env_metrics].mean().mean()
                    trends.append({
                        'category': 'Recycled Materials',
                        'trend': f"Average environmental impact: {avg_env_impact:.2f} (lower is better)",
                        'direction': 'improving' if avg_env_impact < 5 else 'needs attention'
                    })
            
            # If no specific trends found, provide general insights
            if not trends:
                trends = [
                    {
                        'category': 'Sustainability Focus',
                        'trend': 'Growing adoption of organic and recycled materials',
                        'direction': 'improving'
                    },
                    {
                        'category': 'Water Conservation',
                        'trend': 'Materials with lower water consumption gaining popularity',
                        'direction': 'improving'
                    },
                    {
                        'category': 'Carbon Footprint',
                        'trend': 'Reduction in GHG emissions through sustainable practices',
                        'direction': 'improving'
                    }
                ]
            
            return {
                'success': True,
                'trends': trends
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'trends': []
            }

    def get_material_recommendations(self, preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered material recommendations based on preferences"""
        try:
            materials_df = self.database.get_materials_dataframe()
            
            # Calculate scores if not present
            if 'sustainability_score' not in materials_df.columns:
                # Environmental score
                env_metrics = ['water_consumption', 'ghg_emissions', 'land_use']
                available_env_metrics = [m for m in env_metrics if m in materials_df.columns]
                if available_env_metrics:
                    env_scores = []
                    for metric in available_env_metrics:
                        min_val = materials_df[metric].min()
                        max_val = materials_df[metric].max()
                        if max_val > min_val:
                            normalized = 10 * (1 - (materials_df[metric] - min_val) / (max_val - min_val))
                            env_scores.append(normalized)
                        else:
                            env_scores.append(pd.Series(5, index=materials_df.index))
                    
                    if env_scores:
                        materials_df['sustainability_score'] = pd.concat(env_scores, axis=1).mean(axis=1)
            
            if 'cost_score' not in materials_df.columns:
                # Cost score
                if 'raw_material_cost' in materials_df.columns:
                    min_cost = materials_df['raw_material_cost'].min()
                    max_cost = materials_df['raw_material_cost'].max()
                    if max_cost > min_cost:
                        materials_df['cost_score'] = 10 * (1 - (materials_df['raw_material_cost'] - min_cost) / (max_cost - min_cost))
                    else:
                        materials_df['cost_score'] = 5
            
            if 'durability_score' not in materials_df.columns:
                # Durability score
                if 'tensile_strength' in materials_df.columns:
                    min_strength = materials_df['tensile_strength'].min()
                    max_strength = materials_df['tensile_strength'].max()
                    if max_strength > min_strength:
                        materials_df['durability_score'] = 10 * (materials_df['tensile_strength'] - min_strength) / (max_strength - min_strength)
                    else:
                        materials_df['durability_score'] = 5
            
            if 'comfort_score' not in materials_df.columns:
                # Comfort score
                if 'moisture_regain' in materials_df.columns:
                    min_moisture = materials_df['moisture_regain'].min()
                    max_moisture = materials_df['moisture_regain'].max()
                    if max_moisture > min_moisture:
                        materials_df['comfort_score'] = 10 * (materials_df['moisture_regain'] - min_moisture) / (max_moisture - min_moisture)
                    else:
                        materials_df['comfort_score'] = 5
            
            # Filter based on preferences
            priority = preferences.get('priority', 'sustainability')
            budget = preferences.get('budget', 'medium')
            
            # Sort by priority
            if priority == 'sustainability':
                sorted_df = materials_df.sort_values('sustainability_score', ascending=False)
            elif priority == 'cost':
                sorted_df = materials_df.sort_values('cost_score', ascending=False)
            elif priority == 'durability':
                sorted_df = materials_df.sort_values('durability_score', ascending=False)
            elif priority == 'comfort':
                sorted_df = materials_df.sort_values('comfort_score', ascending=False)
            else:
                sorted_df = materials_df
            
            # Get top recommendations
            top_materials = sorted_df.head(5)
            
            recommendations = []
            for _, row in top_materials.iterrows():
                reason = f"High {priority} score ({row.get(f'{priority}_score', 0):.2f}/10)"
                if budget == 'low' and row.get('cost_score', 5) < 5:
                    reason += ", cost-effective"
                elif budget == 'high' and row.get('cost_score', 5) > 7:
                    reason += ", premium quality"
                
                recommendations.append({
                    'name': row['material'],
                    'reason': reason,
                    'sustainability_score': row.get('sustainability_score', 0),
                    'cost_score': row.get('cost_score', 0),
                    'durability_score': row.get('durability_score', 0),
                    'comfort_score': row.get('comfort_score', 0),
                    'source': 'AI Analysis'
                })
            
            return {
                'success': True,
                'materials': recommendations
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'materials': []
            }
