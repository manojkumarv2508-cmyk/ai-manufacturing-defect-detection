import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image

# Import helper modules securely from local files
from utils import predict_image, get_status_color, check_api_health
from history import add_prediction, get_history, clear_history

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Quality Inspection",
    page_icon="🏭",
    layout="wide"
)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### 🏭 Project")
    st.markdown("**Industrial Vision AI**  \n**Quality Inspection System**")
    st.markdown("*Version: 1.0*")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Engine")
    st.markdown("**Architecture:**  \nResNet18 (Transfer Learning)")
    st.markdown("**Framework:**  \nPyTorch")
    st.markdown("**Dataset:**  \nCasting Product Dataset")
    st.markdown("**Classes:**  \nDefective / OK")
    st.markdown("**Validation Accuracy:**  \n99.30%")
    
    st.markdown("---")
    st.markdown("### ⚙ System")
    st.markdown("**Backend:**  \nFastAPI")
    st.markdown("**Frontend:**  \nStreamlit")
    st.markdown("**Visualization:**  \nPlotly")
    st.markdown("**Batch Prediction:**  \nEnabled")
    st.markdown("**CSV Export:**  \nEnabled")
    
    st.markdown("---")
    st.markdown("### 📊 Performance")
    avg_inf_placeholder = st.empty()
    avg_inf_placeholder.markdown("**Average Inference:**  \n*Pending Predictions*")
    st.markdown("**Deployment:**  \nCPU Optimized")
    st.markdown("**Model:**  \ndefect_detector_resnet18.pth")
    
    # API Health Status Card
    st.markdown("---")
    is_healthy = check_api_health()
    if is_healthy:
        st.success("🟢 API Status: Online")
    else:
        st.error("🔴 API Status: Offline")
        
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Manoj Kumar V**")
    st.markdown("*Artificial Intelligence &  \nMachine Learning Engineer*")
    
    st.markdown("---")
    if st.button("Clear Prediction History", use_container_width=True):
        clear_history()
        st.success("History cleared!")

# --- MAIN CONTENT HEADER ---
st.title("Industrial Vision AI Quality Inspection System")
st.subheader("Real-Time Casting Defect Detection using Deep Learning, FastAPI, and Streamlit")
st.markdown("Upload images of casting components below to automatically inspect them for manufacturing defects.")

# --- UPLOAD SECTION ---
uploaded_files = st.file_uploader(
    "Upload Component Images", 
    accept_multiple_files=True, 
    type=["jpg", "jpeg", "png"],
    help="Select multiple images to process them in a batch."
)

if uploaded_files:
    # --- PREVIEW GALLERY ---
    st.markdown("### Image Preview Gallery")
    # Display all uploaded images in responsive columns (4 columns per row)
    num_cols = 4
    cols = st.columns(num_cols)
    for i, file in enumerate(uploaded_files):
        with cols[i % num_cols]:
            image = Image.open(file)
            st.image(image, caption=file.name, use_container_width=True)

    # --- PREDICT ALL BUTTON ---
    st.markdown("---")
    if st.button("Predict All Images", type="primary", use_container_width=True):
        
        # UX elements for loading state
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results_list = []
        
        with st.spinner("Processing images through Deep Learning Pipeline..."):
            total_files = len(uploaded_files)
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name} ({i+1}/{total_files})...")
                
                # Rewind file pointer just in case it was read earlier
                file.seek(0)
                image_bytes = file.read()
                
                # Call FastAPI backend
                result = predict_image(image_bytes, file.name)
                
                if result:
                    # Log to session state history
                    add_prediction(file.name, result)
                    
                    # Determine sort priority (Defective first)
                    is_defective = "def" in result['predicted_class'].lower()
                    
                    results_list.append({
                        "Image Name": file.name,
                        "Prediction": result['predicted_class'],
                        "Confidence (%)": result['confidence_score'] * 100,
                        "Inference Time (ms)": result['inference_time_seconds'] * 1000,
                        "Is Defective": is_defective  # Hidden sorting column
                    })
                
                # Update progress bar
                progress_bar.progress((i + 1) / total_files)
                
        # Clean up loading UX
        status_text.empty()
        
        if results_list:
            st.success(f"Successfully processed {len(results_list)} images!")
            
            # --- RESULTS TABLE ---
            df_results = pd.DataFrame(results_list)
            
            # Sort so defective images appear at the top for immediate attention
            df_results = df_results.sort_values(by="Is Defective", ascending=False).drop(columns=["Is Defective"])
            
            st.markdown("### Prediction Results")
            
            # Style the dataframe colors (Red for defective, Green for OK)
            def highlight_pred(row):
                if "def" in str(row['Prediction']).lower():
                    return ['background-color: rgba(255, 75, 75, 0.15)'] * len(row)
                return ['background-color: rgba(0, 204, 150, 0.15)'] * len(row)
                
            styled_df = df_results.style.apply(highlight_pred, axis=1).format({
                "Inference Time (ms)": "{:.1f} ms"
            })
            
            st.dataframe(
                styled_df, 
                use_container_width=True,
                column_config={
                    "Confidence (%)": st.column_config.ProgressColumn(
                        "Confidence (%)",
                        help="Prediction confidence",
                        format="%.2f%%",
                        min_value=0,
                        max_value=100,
                    )
                }
            )
            
            # --- CSV DOWNLOAD ---
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="defect_detection_results.csv",
                mime="text/csv",
            )
            
            # --- SUMMARY ANALYTICS ---
            st.markdown("### Summary Analytics")
            
            total_processed = len(df_results)
            defective_count = sum(df_results['Prediction'].str.lower().str.contains("def"))
            ok_count = total_processed - defective_count
            defect_rate = (defective_count / total_processed) * 100 if total_processed > 0 else 0
            avg_conf = df_results['Confidence (%)'].mean()
            avg_time = df_results['Inference Time (ms)'].mean()
            
            # Update the sidebar placeholder
            avg_inf_placeholder.markdown(f"**Average Inference:**  \n{avg_time:.1f} ms")
            
            # New metrics
            max_conf = df_results['Confidence (%)'].max()
            min_conf = df_results['Confidence (%)'].min()
            fastest_time = df_results['Inference Time (ms)'].min()
            slowest_time = df_results['Inference Time (ms)'].max()
            
            # Layout metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Images", total_processed)
            col2.metric("Defective Parts", defective_count)
            col3.metric("Accepted Parts", ok_count)
            col4.metric("Defect Rate", f"{defect_rate:.1f}%", "- Optimal" if defect_rate <= 15 else "High", delta_color="inverse")
            
            col5, col6, col7, col8 = st.columns(4)
            col5.metric("Avg Confidence", f"{avg_conf:.2f}%")
            col6.metric("Highest Confidence", f"{max_conf:.2f}%")
            col7.metric("Lowest Confidence", f"{min_conf:.2f}%")
            col8.metric("Inspection Status", "Warning" if defect_rate > 15 else "Optimal", delta_color="off")
            
            col9, col10, col11, col12 = st.columns(4)
            col9.metric("Avg Inference", f"{avg_time:.1f} ms")
            col10.metric("Fastest Pred.", f"{fastest_time:.1f} ms")
            col11.metric("Slowest Pred.", f"{slowest_time:.1f} ms")
            with col12: st.empty()
            
            # --- VISUAL ANALYTICS ---
            st.markdown("### Visual Analytics")
            
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                # 1. Pie Chart: Defective vs OK
                pie_data = pd.DataFrame({
                    "Class": ["Defective", "OK"],
                    "Count": [defective_count, ok_count]
                })
                fig_pie = px.pie(
                    pie_data, 
                    names="Class", 
                    values="Count", 
                    title="Defective vs OK Distribution",
                    color="Class",
                    color_discrete_map={"Defective": "#FF4B4B", "OK": "#00CC96"},
                    hole=0.4
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # 3. Histogram: Confidence distribution
                fig_hist = px.histogram(
                    df_results, 
                    x="Confidence (%)", 
                    title="Confidence Distribution",
                    nbins=20,
                    color_discrete_sequence=["#636EFA"]
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with v_col2:
                # 2. Bar Chart: Confidence per image
                fig_bar = px.bar(
                    df_results, 
                    x="Image Name", 
                    y="Confidence (%)", 
                    color="Prediction",
                    title="Confidence per Image",
                    color_discrete_map={
                        c: get_status_color(c) for c in df_results['Prediction'].unique()
                    }
                )
                st.plotly_chart(fig_bar, use_container_width=True)

# --- PREDICTION HISTORY ---
st.markdown("---")
with st.expander("View Prediction History"):
    history_df = get_history()
    if not history_df.empty:
        # Format the history dataframe before displaying
        styled_hist = history_df.style.format({
            "Confidence": "{:.2%}",
            "Inference Time (s)": "{:.4f} s"
        })
        st.dataframe(styled_hist, use_container_width=True)
    else:
        st.info("No predictions made yet. Upload and process images to see history.")

# --- FOOTER ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding-top: 10px;'>
        <p><strong>Industrial Vision AI Quality Inspection System</strong></p>
        <p>Developed by Manoj Kumar V</p>
        <p><em>Powered by PyTorch • FastAPI • Streamlit • Plotly</em></p>
    </div>
    """,
    unsafe_allow_html=True
)
