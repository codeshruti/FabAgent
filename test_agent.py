#!/usr/bin/env python3
"""
Test script for FabricAgent functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import FabricDatabase
from rag_agent import FabricAgent

def test_fabric_agent():
    """Test the FabricAgent functionality"""
    print("🧪 Testing FabricAgent...")
    
    try:
        # Initialize database
        print("📊 Initializing database...")
        database = FabricDatabase()
        database.init_database()
        
        # Initialize agent
        print("🤖 Initializing FabricAgent...")
        agent = FabricAgent(database)
        
        # Test get_latest_insights
        print("🔍 Testing get_latest_insights...")
        insights = agent.get_latest_insights()
        print(f"✅ Insights: {insights.get('success', False)}")
        if insights.get('materials'):
            print(f"   Found {len(insights['materials'])} materials")
        
        # Test analyze_sustainability_trends
        print("📈 Testing analyze_sustainability_trends...")
        trends = agent.analyze_sustainability_trends()
        print(f"✅ Trends: {trends.get('success', False)}")
        if trends.get('trends'):
            print(f"   Found {len(trends['trends'])} trends")
        
        # Test get_material_recommendations
        print("🎯 Testing get_material_recommendations...")
        preferences = {
            'priority': 'sustainability',
            'budget': 'medium'
        }
        recommendations = agent.get_material_recommendations(preferences)
        print(f"✅ Recommendations: {recommendations.get('success', False)}")
        if recommendations.get('materials'):
            print(f"   Found {len(recommendations['materials'])} recommendations")
            for i, material in enumerate(recommendations['materials'][:3], 1):
                print(f"   {i}. {material['name']}: {material['reason']}")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_fabric_agent()
    sys.exit(0 if success else 1) 