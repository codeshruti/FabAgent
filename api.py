import os
import sys
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import FabricDatabase
from optimizer import optimize_fabric_blend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FabAgent API",
    description="LLM-Based Agentic Optimization Framework for Design of Sustainable Fabrics",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for database
database = None

def get_database():
    global database
    if database is None:
        database = FabricDatabase()
        database.init_database()
    return database

# Pydantic models for request/response
class MaterialSearchRequest(BaseModel):
    query: str
    filters: Optional[Dict[str, Any]] = None

class MaterialSearchResponse(BaseModel):
    success: bool
    materials: List[Dict[str, Any]]
    total_count: int
    query: str

class BlendOptimizationRequest(BaseModel):
    max_materials: int = 5
    population_size: int = 100
    n_generations: int = 50
    sustainability_weight: float = 0.3
    durability_weight: float = 0.3
    comfort_weight: float = 0.2
    cost_weight: float = 0.2
    constraints: Optional[Dict[str, Any]] = None

class BlendOptimizationResponse(BaseModel):
    success: bool
    solutions: List[Dict[str, Any]]
    best_solution: Optional[Dict[str, Any]]
    optimization_metrics: Dict[str, Any]

class BlendAnalysisRequest(BaseModel):
    materials: List[str]
    proportions: List[float]
    analysis_type: str = "comprehensive"  # comprehensive, sustainability, cost, performance

class BlendAnalysisResponse(BaseModel):
    success: bool
    analysis: Dict[str, Any]
    scores: Dict[str, float]
    recommendations: List[str]

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "FabAgent API - LLM-Based Agentic Optimization Framework",
        "version": "1.0.0",
        "endpoints": {
            "materials": "/api/materials/search",
            "blends": "/api/blends/optimize",
            "analysis": "/api/blends/analyze",
            "categories": "/api/materials/categories",
            "metrics": "/api/materials/metrics",
            "insights": "/api/insights/latest"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        db = get_database()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

# Material Search Endpoints

@app.post("/api/materials/search", response_model=MaterialSearchResponse)
async def search_materials(request: MaterialSearchRequest):
    """Search materials using database"""
    try:
        db = get_database()
        
        # Database search
        results = db.search_materials(request.query)
        materials = []
        for result in results:
            # Get all materials and find the matching one with metrics
            all_materials = db.get_all_materials()
            material_with_metrics = None
            for material in all_materials:
                if material['id'] == result['id']:
                    material_with_metrics = material
                    break
            
            if material_with_metrics:
                materials.append(material_with_metrics)
            else:
                # Fallback to basic result without metrics
                materials.append({
                    "id": result["id"],
                    "name": result["name"],
                    "category": result["category"],
                    "description": result["description"],
                    "metrics": {}
                })
        
        return MaterialSearchResponse(
            success=True,
            materials=materials,
            total_count=len(materials),
            query=request.query
        )
            
    except Exception as e:
        logger.error(f"Material search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Material search failed: {str(e)}")

@app.get("/api/materials/{material_id}")
async def get_material_details(material_id: int):
    """Get detailed information about a specific material"""
    try:
        db = get_database()
        
        # Get all materials and find the one with matching ID
        all_materials = db.get_all_materials()
        material = None
        for mat in all_materials:
            if mat['id'] == material_id:
                material = mat
                break
        
        if not material:
            raise HTTPException(status_code=404, detail="Material not found")
        
        return {
            "success": True,
            "material": {
                "id": material["id"],
                "name": material["name"],
                "category": material["category"],
                "description": material["description"],
                "metrics": material["metrics"]
            }
        }
    
    except Exception as e:
        logger.error(f"Get material details failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get material details failed: {str(e)}")

# Blend Optimization Endpoints

@app.post("/api/blends/optimize", response_model=BlendOptimizationResponse)
async def optimize_blend(request: BlendOptimizationRequest):
    """Optimize fabric blend using NSGA-II algorithm"""
    try:
        db = get_database()
        materials_df = db.get_materials_dataframe()
        
        if materials_df.empty:
            raise HTTPException(status_code=400, detail="No materials data available for optimization")
        
        # Run optimization
        results = optimize_fabric_blend(
            materials_df=materials_df,
            max_materials=request.max_materials,
            population_size=request.population_size,
            n_generations=request.n_generations
        )
        
        if results and 'solutions' in results:
            # Calculate optimization metrics
            optimization_metrics = {
                "total_solutions": len(results['solutions']),
                "generations": request.n_generations,
                "population_size": request.population_size,
                "max_materials": request.max_materials
            }
            
            return BlendOptimizationResponse(
                success=True,
                solutions=results['solutions'],
                best_solution=results.get('best_solution'),
                optimization_metrics=optimization_metrics
            )
        else:
            raise HTTPException(status_code=500, detail="Optimization failed to produce results")
    
    except Exception as e:
        logger.error(f"Blend optimization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blend optimization failed: {str(e)}")

@app.post("/api/blends/analyze", response_model=BlendAnalysisResponse)
async def analyze_blend(request: BlendAnalysisRequest):
    """Analyze a specific blend composition"""
    try:
        agent = get_agent()
        
        # Create blend data
        blend_data = {
            'materials': request.materials,
            'ratios': request.proportions
        }
        
        # Analyze blend performance
        performance = agent.analyze_blend_performance(blend_data)
        
        if 'performance' in performance:
            scores = performance['performance']
            recommendations = performance.get('recommendations', [])
            
            analysis = {
                "blend_composition": blend_data,
                "overall_score": performance.get('overall_score', 0),
                "detailed_analysis": performance
            }
            
            return BlendAnalysisResponse(
                success=True,
                analysis=analysis,
                scores=scores,
                recommendations=recommendations
            )
        else:
            raise HTTPException(status_code=500, detail="Blend analysis failed")
    
    except Exception as e:
        logger.error(f"Blend analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Blend analysis failed: {str(e)}")

# Additional utility endpoints

@app.get("/api/materials/categories")
async def get_material_categories():
    """Get all available material categories"""
    try:
        db = get_database()
        categories = db.get_material_categories()
        return {
            "success": True,
            "categories": categories
        }
    except Exception as e:
        logger.error(f"Get categories failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get categories failed: {str(e)}")

@app.get("/api/materials/metrics")
async def get_available_metrics():
    """Get all available material metrics"""
    try:
        db = get_database()
        metrics = db.get_metric_categories()
        return {
            "success": True,
            "metrics": metrics
        }
    except Exception as e:
        logger.error(f"Get metrics failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get metrics failed: {str(e)}")

@app.get("/api/insights/latest")
async def get_latest_insights():
    """Get latest material discoveries and insights"""
    try:
        db = get_database()
        materials = db.get_all_materials()
        
        # Generate simple insights based on available data
        insights = []
        if materials:
            # Find materials with best sustainability scores
            sustainability_scores = []
            for material in materials:
                if 'metrics' in material and 'ghg_emissions' in material['metrics']:
                    sustainability_scores.append({
                        'name': material['name'],
                        'ghg_emissions': material['metrics']['ghg_emissions']['value']
                    })
            
            if sustainability_scores:
                best_sustainable = min(sustainability_scores, key=lambda x: x['ghg_emissions'])
                insights.append(f"Best sustainable material: {best_sustainable['name']} with {best_sustainable['ghg_emissions']} kg CO2e/kg")
        
        return {
            "success": True,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Get latest insights failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Get latest insights failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 