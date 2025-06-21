# FabAgent FastAPI Backend

## Overview

The FabAgent FastAPI backend provides a comprehensive REST API for the LLM-Based Agentic Optimization Framework for Design of Sustainable Fabrics. This API implements all the features described in the research paper, including RAG-powered material search, blend optimization, and agentic recommendations.

## 🚀 Quick Start

### Running the API

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=your_api_key_here

# Run the API
python api.py
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Documentation**: http://localhost:8000/docs
- **Alternative Documentation**: http://localhost:8000/redoc

### Using Docker

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run directly with Docker
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key fabagent
```

## 📚 API Endpoints

### Core Endpoints

#### 1. Health Check
```http
GET /health
```
Returns the health status of the API and its components.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "agent": "initialized",
  "timestamp": "2024-01-01T12:00:00"
}
```

#### 2. Root Information
```http
GET /
```
Returns API information and available endpoints.

### Material Search Endpoints

#### 3. Search Materials
```http
POST /api/materials/search
```

**Request Body:**
```json
{
  "query": "sustainable cotton",
  "search_type": "ai_enhanced",
  "filters": {
    "category": "natural_fibers",
    "sustainability_score": 7.0
  }
}
```

**Search Types:**
- `database`: Traditional database search
- `ai_enhanced`: RAG-powered search with AI insights
- `web_research`: Pure web research results

**Response:**
```json
{
  "success": true,
  "materials": [
    {
      "id": 1,
      "name": "Organic Cotton",
      "category": "natural_fibers",
      "description": "Sustainably grown cotton...",
      "metrics": {
        "sustainability_score": 8.5,
        "durability_score": 7.2,
        "comfort_score": 8.8,
        "cost_score": 6.5
      },
      "source": "database"
    }
  ],
  "total_count": 1,
  "search_type": "ai_enhanced",
  "query": "sustainable cotton"
}
```

#### 4. Get Material Details
```http
GET /api/materials/{material_id}
```

**Response:**
```json
{
  "success": true,
  "material": {
    "id": 1,
    "name": "Organic Cotton",
    "category": "natural_fibers",
    "description": "Detailed description...",
    "metrics": {
      "sustainability_score": 8.5,
      "durability_score": 7.2,
      "comfort_score": 8.8,
      "cost_score": 6.5
    }
  }
}
```

### Blend Optimization Endpoints

#### 5. Optimize Blend
```http
POST /api/blends/optimize
```

**Request Body:**
```json
{
  "max_materials": 5,
  "population_size": 100,
  "n_generations": 50,
  "sustainability_weight": 0.3,
  "durability_weight": 0.3,
  "comfort_weight": 0.2,
  "cost_weight": 0.2,
  "constraints": {
    "min_sustainability": 7.0,
    "max_cost": 8.0
  }
}
```

**Response:**
```json
{
  "success": true,
  "solutions": [
    {
      "materials": ["Organic Cotton", "Hemp", "Bamboo"],
      "proportions": [0.5, 0.3, 0.2],
      "sustainability_score": 8.2,
      "durability_score": 7.8,
      "comfort_score": 8.5,
      "cost_score": 7.1,
      "overall_score": 8.0
    }
  ],
  "best_solution": {
    "materials": ["Organic Cotton", "Hemp", "Bamboo"],
    "proportions": [0.5, 0.3, 0.2],
    "overall_score": 8.0
  },
  "optimization_metrics": {
    "total_solutions": 100,
    "generations": 50,
    "population_size": 100,
    "max_materials": 5
  }
}
```

#### 6. Analyze Blend
```http
POST /api/blends/analyze
```

**Request Body:**
```json
{
  "materials": ["Organic Cotton", "Hemp", "Bamboo"],
  "proportions": [0.5, 0.3, 0.2],
  "analysis_type": "comprehensive"
}
```

**Response:**
```json
{
  "success": true,
  "analysis": {
    "blend_composition": {
      "materials": ["Organic Cotton", "Hemp", "Bamboo"],
      "ratios": [0.5, 0.3, 0.2]
    },
    "overall_score": 8.0,
    "detailed_analysis": {
      "performance": {
        "sustainability_score": 8.2,
        "durability_score": 7.8,
        "comfort_score": 8.5,
        "cost_score": 7.1
      }
    }
  },
  "scores": {
    "sustainability_score": 8.2,
    "durability_score": 7.8,
    "comfort_score": 8.5,
    "cost_score": 7.1
  },
  "recommendations": [
    "Consider increasing hemp proportion for better durability",
    "This blend offers excellent sustainability performance"
  ]
}
```

### Agent Endpoints

#### 7. Get AI Recommendations
```http
POST /api/agent/recommend
```

**Request Body:**
```json
{
  "preferences": {
    "priority": "sustainability",
    "budget": "medium",
    "application": "casual_wear",
    "durability_requirement": "high"
  },
  "context": "Looking for sustainable materials for outdoor clothing",
  "recommendation_type": "materials"
}
```

**Recommendation Types:**
- `materials`: Material recommendations
- `blends`: Blend suggestions
- `optimization`: Optimization strategies

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "name": "Organic Hemp",
      "reasoning": "High sustainability score with excellent durability",
      "sustainability_score": 9.2,
      "durability_score": 8.8
    }
  ],
  "reasoning": "AI analysis of material properties and user preferences",
  "confidence": 0.85
}
```

#### 8. Extract Data
```http
POST /api/agent/extract
```

**Request Body:**
```json
{
  "source": "scientific_paper",
  "content": "Lyocell fiber exhibits excellent moisture absorption...",
  "extraction_type": "material_properties"
}
```

**Response:**
```json
{
  "success": true,
  "extracted_data": {
    "material_name": "Lyocell",
    "properties": {
      "moisture_absorption": "excellent",
      "sustainability": "high"
    }
  },
  "confidence": 0.85,
  "source": "scientific_paper"
}
```

#### 9. Analyze Trends
```http
POST /api/agent/trends
```

**Request Body:**
```json
{
  "analysis_type": "sustainability",
  "time_period": "2023-2024"
}
```

**Response:**
```json
{
  "success": true,
  "trends": [
    {
      "category": "Sustainable Materials",
      "trend": "Growing adoption",
      "direction": "improving"
    }
  ],
  "insights": [
    "Growing adoption of sustainable materials",
    "Reduction in environmental impact across categories"
  ],
  "recommendations": [
    "Consider recycled materials for new blends",
    "Focus on materials with lower water consumption"
  ]
}
```

### Utility Endpoints

#### 10. Get Material Categories
```http
GET /api/materials/categories
```

#### 11. Get Available Metrics
```http
GET /api/materials/metrics
```

#### 12. Get Latest Insights
```http
GET /api/insights/latest
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for AI features
- `DATABASE_PATH`: Path to the SQLite database (default: `data/fabrics.db`)

### API Configuration

The API supports the following configuration options:

- **CORS**: Enabled for all origins (configurable)
- **Logging**: Structured logging with different levels
- **Rate Limiting**: Can be configured if needed
- **Authentication**: Can be added for production use

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_api.py
```

This will test all endpoints and verify the RAG functionality.

## 📊 API Documentation

### Interactive Documentation

Visit http://localhost:8000/docs for interactive API documentation powered by Swagger UI.

### Alternative Documentation

Visit http://localhost:8000/redoc for alternative documentation format.

## 🔍 RAG Features

The API implements several RAG (Retrieval-Augmented Generation) features:

1. **AI-Enhanced Search**: Combines database search with AI insights
2. **Web Research**: Extracts information from web sources
3. **Data Extraction**: Extracts material properties from various sources
4. **Trend Analysis**: Analyzes sustainability and performance trends
5. **Intelligent Recommendations**: Provides context-aware recommendations

## 🚀 Deployment

### Production Deployment

For production deployment, consider:

1. **Environment Variables**: Set proper environment variables
2. **Database**: Use a production database (PostgreSQL, MySQL)
3. **Authentication**: Add API key or OAuth authentication
4. **Rate Limiting**: Implement rate limiting
5. **Monitoring**: Add health checks and monitoring
6. **SSL/TLS**: Use HTTPS in production

### Docker Deployment

```bash
# Build production image
docker build -t fabagent-api .

# Run with environment variables
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  -e DATABASE_PATH=/app/data/production.db \
  fabagent-api
```

### Kubernetes Deployment

Create a Kubernetes deployment with proper resource limits and health checks.

## 📈 Performance

The API is optimized for:

- **Fast Response Times**: Caching and efficient database queries
- **Scalability**: Stateless design for horizontal scaling
- **Reliability**: Comprehensive error handling and logging
- **Flexibility**: Modular design for easy extension

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For support and questions:

1. Check the API documentation at `/docs`
2. Review the test examples in `test_api.py`
3. Check the logs for error details
4. Open an issue on GitHub

---

**FabAgent API** - Empowering sustainable fabric design with AI-driven insights and optimization. 