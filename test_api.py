#!/usr/bin/env python3
"""
Test script for FabAgent FastAPI endpoints
"""

import requests
import json
import time
import os
from typing import Dict, Any

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_material_search():
    """Test material search endpoints"""
    print("\n🔍 Testing material search...")
    
    # Test database search
    print("  Testing database search...")
    search_data = {
        "query": "cotton",
        "search_type": "database"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/materials/search", json=search_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Database search successful: Found {data['total_count']} materials")
        else:
            print(f"❌ Database search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Database search error: {str(e)}")
    
    # Test AI-enhanced search
    print("  Testing AI-enhanced search...")
    search_data = {
        "query": "sustainable materials",
        "search_type": "ai_enhanced"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/materials/search", json=search_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ AI-enhanced search successful: Found {data['total_count']} results")
        else:
            print(f"❌ AI-enhanced search failed: {response.status_code}")
    except Exception as e:
        print(f"❌ AI-enhanced search error: {str(e)}")

def test_blend_optimization():
    """Test blend optimization endpoint"""
    print("\n🔍 Testing blend optimization...")
    
    optimization_data = {
        "max_materials": 3,
        "population_size": 50,
        "n_generations": 20,
        "sustainability_weight": 0.4,
        "durability_weight": 0.3,
        "comfort_weight": 0.2,
        "cost_weight": 0.1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/blends/optimize", json=optimization_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blend optimization successful: Generated {len(data['solutions'])} solutions")
            if data['best_solution']:
                print(f"   Best solution score: {data['best_solution'].get('overall_score', 'N/A')}")
        else:
            print(f"❌ Blend optimization failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blend optimization error: {str(e)}")

def test_blend_analysis():
    """Test blend analysis endpoint"""
    print("\n🔍 Testing blend analysis...")
    
    analysis_data = {
        "materials": ["Organic Cotton", "Hemp", "Bamboo"],
        "proportions": [0.5, 0.3, 0.2],
        "analysis_type": "comprehensive"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/blends/analyze", json=analysis_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blend analysis successful")
            print(f"   Overall score: {data['analysis']['overall_score']}")
            print(f"   Recommendations: {len(data['recommendations'])} suggestions")
        else:
            print(f"❌ Blend analysis failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blend analysis error: {str(e)}")

def test_agent_recommendations():
    """Test agent recommendation endpoints"""
    print("\n🔍 Testing agent recommendations...")
    
    # Test material recommendations
    print("  Testing material recommendations...")
    recommendation_data = {
        "preferences": {
            "priority": "sustainability",
            "budget": "medium",
            "application": "casual_wear"
        },
        "recommendation_type": "materials"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/recommend", json=recommendation_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Material recommendations successful: {len(data['recommendations'])} recommendations")
            print(f"   Confidence: {data['confidence']}")
        else:
            print(f"❌ Material recommendations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Material recommendations error: {str(e)}")
    
    # Test blend recommendations
    print("  Testing blend recommendations...")
    recommendation_data["recommendation_type"] = "blends"
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/recommend", json=recommendation_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Blend recommendations successful: {len(data['recommendations'])} recommendations")
        else:
            print(f"❌ Blend recommendations failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Blend recommendations error: {str(e)}")

def test_data_extraction():
    """Test data extraction endpoint"""
    print("\n🔍 Testing data extraction...")
    
    extraction_data = {
        "source": "user_input",
        "content": "Lyocell is a sustainable fiber made from wood pulp with excellent moisture absorption and breathability properties.",
        "extraction_type": "material_properties"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/extract", json=extraction_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data extraction successful")
            print(f"   Confidence: {data['confidence']}")
            print(f"   Source: {data['source']}")
        else:
            print(f"❌ Data extraction failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data extraction error: {str(e)}")

def test_trend_analysis():
    """Test trend analysis endpoint"""
    print("\n🔍 Testing trend analysis...")
    
    # Test sustainability trends
    print("  Testing sustainability trends...")
    trend_data = {
        "analysis_type": "sustainability"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/trends", json=trend_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sustainability trends successful: {len(data['trends'])} trends")
            print(f"   Insights: {len(data['insights'])} insights")
        else:
            print(f"❌ Sustainability trends failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Sustainability trends error: {str(e)}")
    
    # Test cost trends
    print("  Testing cost trends...")
    trend_data["analysis_type"] = "cost"
    
    try:
        response = requests.post(f"{BASE_URL}/api/agent/trends", json=trend_data)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Cost trends successful: {len(data['trends'])} trends")
        else:
            print(f"❌ Cost trends failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Cost trends error: {str(e)}")

def test_utility_endpoints():
    """Test utility endpoints"""
    print("\n🔍 Testing utility endpoints...")
    
    # Test categories endpoint
    print("  Testing material categories...")
    try:
        response = requests.get(f"{BASE_URL}/api/materials/categories")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Categories successful: {len(data['categories'])} categories")
        else:
            print(f"❌ Categories failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Categories error: {str(e)}")
    
    # Test metrics endpoint
    print("  Testing available metrics...")
    try:
        response = requests.get(f"{BASE_URL}/api/materials/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics successful: {len(data['metrics'])} metric categories")
        else:
            print(f"❌ Metrics failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Metrics error: {str(e)}")
    
    # Test latest insights
    print("  Testing latest insights...")
    try:
        response = requests.get(f"{BASE_URL}/api/insights/latest")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Latest insights successful: {len(data['insights'])} insights")
        else:
            print(f"❌ Latest insights failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Latest insights error: {str(e)}")

def main():
    """Run all API tests"""
    print("🚀 Starting FabAgent API Tests")
    print("=" * 50)
    
    # Check if API is running
    print("📡 Checking if API is running...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✅ API is running")
            data = response.json()
            print(f"   API Version: {data['version']}")
            print(f"   Available endpoints: {len(data['endpoints'])}")
        else:
            print(f"❌ API not responding: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {str(e)}")
        print("   Make sure the API is running on http://localhost:8000")
        return
    
    # Run all tests
    test_health_check()
    test_material_search()
    test_blend_optimization()
    test_blend_analysis()
    test_agent_recommendations()
    test_data_extraction()
    test_trend_analysis()
    test_utility_endpoints()
    
    print("\n" + "=" * 50)
    print("🎉 API testing completed!")

if __name__ == "__main__":
    main() 