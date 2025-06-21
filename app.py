import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from typing import List, Dict, Any, Optional
import os
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import openai

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import FabricDatabase
from optimizer import optimize_fabric_blend, compare_with_commercial_blends

st.set_page_config(
    page_title="FabAgent - Sustainable Fabric Design",
    page_icon="🧵",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .ai-insight {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_database():
    db = FabricDatabase()
    db.init_database()
    return db

def calculate_sustainability_score(row):
    if 'water_consumption' in row and 'ghg_emissions' in row and 'land_use' in row:
        water_score = max(0, 10 - (row['water_consumption'] / 1000))
        ghg_score = max(0, 10 - (row['ghg_emissions'] / 2))
        land_score = max(0, 10 - (row['land_use'] / 1))
        return (water_score + ghg_score + land_score) / 3
    return 5.0

def calculate_cost_score(row):
    if 'raw_material_cost' in row:
        return max(0, 10 - (row['raw_material_cost'] / 2))
    return 5.0

def calculate_durability_score(row):
    if 'tensile_strength' in row:
        return min(10, row['tensile_strength'] / 100)
    return 5.0

def main():
    database = load_database()
    
    st.markdown('<h1 class="main-header">🧵 FabAgent</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">LLM-Based Agentic Optimization Framework for Design of Sustainable Fabrics</p>', unsafe_allow_html=True)
    
    sidebar = st.sidebar
    sidebar.title("Navigation")
    
    page = sidebar.radio("Go to:", [
        "📊 Dashboard",
        "🔍 Material Explorer", 
        "⚡ Blend Optimizer",
        "📈 Analytics"
    ])
    
    if page == "📊 Dashboard":
        show_dashboard(database)
    elif page == "🔍 Material Explorer":
        show_material_explorer(database)
    elif page == "⚡ Blend Optimizer":
        show_blend_optimizer(database)
    elif page == "📈 Analytics":
        show_analytics(database)

def show_dashboard(database):
    st.header("📊 Dashboard")
    
    materials_df = database.get_materials_dataframe()
    
    materials_df['sustainability_score'] = materials_df.apply(calculate_sustainability_score, axis=1)
    materials_df['cost_score'] = materials_df.apply(calculate_cost_score, axis=1)
    materials_df['durability_score'] = materials_df.apply(calculate_durability_score, axis=1)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Materials", len(materials_df))
    with col2:
        avg_sustainability = materials_df['sustainability_score'].mean()
        st.metric("Avg Sustainability", f"{avg_sustainability:.2f}")
    with col3:
        avg_cost = materials_df['cost_score'].mean()
        st.metric("Avg Cost Score", f"{avg_cost:.2f}")
    with col4:
        avg_durability = materials_df['durability_score'].mean()
        st.metric("Avg Durability", f"{avg_durability:.2f}")
    
    st.subheader("📈 Material Distribution by Category")
    category_counts = materials_df['category'].value_counts()
    fig = px.pie(values=category_counts.values, names=category_counts.index, title="Material Categories")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🤖 AI Recommendations")
    col1, col2, col3 = st.columns(3)
    with col1:
        priority = st.selectbox("Priority:", ["Sustainability", "Cost", "Durability", "Comfort"])
    with col2:
        budget = st.selectbox("Budget:", ["Low", "Medium", "High"])
    with col3:
        if st.button("🎯 Get AI Recommendations", use_container_width=True):
            try:
                openai_api_key = os.environ.get("OPENAI_API_KEY", "")
                if not openai_api_key:
                    st.error("OPENAI_API_KEY not set. Please set your OpenAI API key as an environment variable.")
                else:
                    client = openai.OpenAI(api_key=openai_api_key)
                    prompt = f"""
You are an expert in sustainable fabric design. Given the following user preferences:

- Priority: {priority}
- Budget: {budget}

Recommend 3 fabric materials or blends, and explain why each is a good fit for these preferences. Focus on sustainability, cost-effectiveness, and practical applications.
"""
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are an expert in sustainable fabric design and material science."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        temperature=0.7
                    )
                    
                    recommendations = response.choices[0].message.content
                    st.success("🤖 AI Recommendations Generated!")
                    st.write(recommendations)
                    
            except Exception as e:
                st.error(f"Error generating AI recommendations: {str(e)}")
                st.info("Make sure you have a valid OpenAI API key and internet connection.")

    if 'gpt_recommendations' in st.session_state and st.session_state.gpt_recommendations:
        st.subheader("🤖 Previous AI Recommendations")
        st.write(st.session_state.gpt_recommendations)

def show_material_explorer(database):
    st.header("🔍 Material Explorer")
    
    st.subheader("Search Materials")
    
    search_query = st.text_input("Search materials by name, category, description, or properties:")
    
    if st.button("🔍 Search", use_container_width=True):
        if search_query:
            with st.spinner("Searching materials..."):
                results = database.search_materials(search_query)
                if results:
                    st.success(f"Found {len(results)} materials")
                    for material in results:
                        with st.expander(f"📦 {material['name']} ({material['category']})"):
                            st.write(f"**Description:** {material['description']}")
                            st.write(f"**Category:** {material['category']}")
                else:
                    st.info("No materials found matching your search.")
        else:
            st.warning("Please enter a search query.")
    
    st.subheader("📋 All Materials")
    materials_df = database.get_materials_dataframe()
    
    if not materials_df.empty:
        materials_df['sustainability_score'] = materials_df.apply(calculate_sustainability_score, axis=1)
        materials_df['cost_score'] = materials_df.apply(calculate_cost_score, axis=1)
        materials_df['durability_score'] = materials_df.apply(calculate_durability_score, axis=1)
        
        st.dataframe(
            materials_df[['material', 'category', 'sustainability_score', 'cost_score', 'durability_score']].round(2),
            use_container_width=True
        )
    else:
        st.info("No materials found in the database.")

def show_blend_optimizer(database):
    st.header("⚡ Blend Optimizer")
    
    st.subheader("Optimization Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_materials = st.slider("Maximum Materials in Blend", 2, 5, 3)
        population_size = st.slider("Population Size", 50, 200, 100)
        n_generations = st.slider("Number of Generations", 20, 100, 50)
    
    with col2:
        st.write("**Optimization Objectives:**")
        st.write("• Environmental Impact (minimize)")
        st.write("• Cost (minimize)")
        st.write("• Durability (maximize)")
        st.write("• Comfort (maximize)")
        
        # Show parameter sensitivity
        if st.checkbox("Show parameter sensitivity analysis"):
            st.write("**Parameter Impact:**")
            st.write(f"• Max Materials: {max_materials} (affects blend complexity)")
            st.write(f"• Population Size: {population_size} (affects search space)")
            st.write(f"• Generations: {n_generations} (affects convergence)")
    
    # Add a quick test button to show parameter sensitivity
    if st.button("🧪 Quick Parameter Test", use_container_width=True):
        with st.spinner("Testing parameter sensitivity..."):
            try:
                materials_df = database.get_materials_dataframe()
                if not materials_df.empty:
                    materials_df['sustainability_score'] = materials_df.apply(calculate_sustainability_score, axis=1)
                    materials_df['cost_score'] = materials_df.apply(calculate_cost_score, axis=1)
                    materials_df['durability_score'] = materials_df.apply(calculate_durability_score, axis=1)
                    
                    # Test with different parameters
                    test_params = [
                        (2, 50, 20),
                        (3, 100, 50),
                        (4, 150, 80)
                    ]
                    
                    test_results = []
                    for test_max, test_pop, test_gen in test_params:
                        result = optimize_fabric_blend(
                            materials_df, 
                            max_materials=test_max,
                            population_size=test_pop,
                            n_generations=test_gen
                        )
                        if result and 'best_solution' in result:
                            best = result['best_solution']
                            test_results.append({
                                'Parameters': f"{test_max} materials, {test_pop} pop, {test_gen} gen",
                                'Materials': ', '.join(best['material_proportions'].keys()),
                                'Environmental': f"{best['environmental_score']:.2f}",
                                'Cost': f"{best['cost_score']:.2f}",
                                'Durability': f"{best['durability_score']:.2f}",
                                'Comfort': f"{best['comfort_score']:.2f}"
                            })
                    
                    if test_results:
                        st.subheader("🧪 Parameter Sensitivity Results")
                        test_df = pd.DataFrame(test_results)
                        st.dataframe(test_df, use_container_width=True)
                        
            except Exception as e:
                st.error(f"Parameter test failed: {str(e)}")

    if st.button("🚀 Optimize Blend", use_container_width=True):
        with st.spinner("Optimizing blend..."):
            try:
                materials_df = database.get_materials_dataframe()
                
                if materials_df.empty:
                    st.error("No materials available for optimization.")
                    return
                
                materials_df['sustainability_score'] = materials_df.apply(calculate_sustainability_score, axis=1)
                materials_df['cost_score'] = materials_df.apply(calculate_cost_score, axis=1)
                materials_df['durability_score'] = materials_df.apply(calculate_durability_score, axis=1)
                
                # Show optimization parameters
                st.info(f"🔧 Optimization Parameters: Max Materials={max_materials}, Population={population_size}, Generations={n_generations}")
                
                result = optimize_fabric_blend(
                    materials_df, 
                    max_materials=max_materials,
                    population_size=population_size,
                    n_generations=n_generations
                )
                
                if result and 'best_solution' in result:
                    st.success("🎉 Blend optimization completed!")
                    
                    best_solution = result['best_solution']
                    
                    st.subheader("🏆 Best Blend Solution")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Materials and Proportions:**")
                        for material, proportion in best_solution['material_proportions'].items():
                            st.write(f"- {material}: {proportion:.1f}%")
                    
                    with col2:
                        st.write("**Performance Scores:**")
                        st.write(f"• Environmental: {best_solution['environmental_score']:.2f}")
                        st.write(f"• Cost: {best_solution['cost_score']:.2f}")
                        st.write(f"• Durability: {best_solution['durability_score']:.2f}")
                        st.write(f"• Comfort: {best_solution['comfort_score']:.2f}")
                    
                    if 'solutions' in result and len(result['solutions']) > 1:
                        st.subheader("📊 All Solutions")
                        
                        solutions_data = []
                        for i, solution in enumerate(result['solutions'][:10]):
                            materials_str = ', '.join([f"{mat} ({prop:.1f}%)" for mat, prop in solution['material_proportions'].items()])
                            solutions_data.append({
                                'Solution': f"Solution {i+1}",
                                'Materials': materials_str,
                                'Environmental': f"{solution['environmental_score']:.2f}",
                                'Cost': f"{solution['cost_score']:.2f}",
                                'Durability': f"{solution['durability_score']:.2f}",
                                'Comfort': f"{solution['comfort_score']:.2f}"
                            })
                        
                        solutions_df = pd.DataFrame(solutions_data)
                        st.dataframe(solutions_df, use_container_width=True)
                
                else:
                    st.error("Optimization failed. Please try different parameters.")
                    
            except Exception as e:
                st.error(f"Error during optimization: {str(e)}")

def show_analytics(database):
    st.header("📈 Analytics")
    
    materials_df = database.get_materials_dataframe()
    
    if materials_df.empty:
        st.info("No materials available for analytics.")
        return
    
    materials_df['sustainability_score'] = materials_df.apply(calculate_sustainability_score, axis=1)
    materials_df['cost_score'] = materials_df.apply(calculate_cost_score, axis=1)
    materials_df['durability_score'] = materials_df.apply(calculate_durability_score, axis=1)
    
    st.subheader("📊 Material Performance Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.scatter(
            materials_df, 
            x='sustainability_score', 
            y='cost_score',
            hover_data=['material', 'category'],
            title="Sustainability vs Cost"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            materials_df, 
            x='durability_score', 
            y='sustainability_score',
            hover_data=['material', 'category'],
            title="Durability vs Sustainability"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Performance Rankings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Top Sustainability Materials:**")
        top_sustainability = materials_df.nlargest(5, 'sustainability_score')[['material', 'sustainability_score']]
        st.dataframe(top_sustainability.round(2), use_container_width=True)
    
    with col2:
        st.write("**Top Cost-Effective Materials:**")
        top_cost = materials_df.nlargest(5, 'cost_score')[['material', 'cost_score']]
        st.dataframe(top_cost.round(2), use_container_width=True)
    
    with col3:
        st.write("**Top Durable Materials:**")
        top_durability = materials_df.nlargest(5, 'durability_score')[['material', 'durability_score']]
        st.dataframe(top_durability.round(2), use_container_width=True)
    
    st.subheader("📈 Category Performance")
    
    category_performance = materials_df.groupby('category').agg({
        'sustainability_score': 'mean',
        'cost_score': 'mean',
        'durability_score': 'mean'
    }).round(2)
    
    fig = px.bar(
        category_performance,
        title="Average Performance by Category",
        barmode='group'
    )
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main() 