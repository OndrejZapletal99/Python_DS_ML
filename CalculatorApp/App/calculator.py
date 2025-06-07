import streamlit as st
import pandas as pd
from joblib import load

# --- Model path setting ---
model_path = "final_model_gbm.joblib"


# Load the trained model
model = load(model_path)

# --- Dictionaries and options definition ---
category_names = {
    "Industrial": ["Fire-rated", "Soundproof", "Thermal-insulated"],
    "Residential": ["Interior", "Exterior", "Balcony"],
    "Commercial": ["Office", "Retail", "Warehouse"],
    "Security": ["Reinforced", "Bulletproof", "Access-controlled"],
    "Design": ["Glass-panel", "Minimalist", "Classic"]
}

type_options = ['Door', 'Frame']
handle_options = ['Aluminum (AL)', 'Steel (ST)', 'Plastic (PL)']
color_options = ['RAL7024', 'RAL7035', 'RAL9005']
hinge_options = ['Left', 'Right']
packing_options = ['Wooden box', 'Cardbox']

# --- Columns for DataFrame as required by the model ---
target_columns = [
    'Width', 'Height', 'Product_batch', 'Product_costs',
    'Product_type_Door', 'Product_type_Frame',
    'Product_category_Commercial', 'Product_category_Design',
    'Product_category_Industrial', 'Product_category_Residential',
    'Product_category_Security', 'Product_subcategory_Access-controlled',
    'Product_subcategory_Balcony', 'Product_subcategory_Bulletproof',
    'Product_subcategory_Classic', 'Product_subcategory_Exterior',
    'Product_subcategory_Fire-rated', 'Product_subcategory_Glass-panel',
    'Product_subcategory_Interior', 'Product_subcategory_Minimalist',
    'Product_subcategory_Office', 'Product_subcategory_Reinforced',
    'Product_subcategory_Retail', 'Product_subcategory_Soundproof',
    'Product_subcategory_Thermal-insulated',
    'Product_subcategory_Warehouse', 'Product_color_RAL7024',
    'Product_color_RAL7035', 'Product_color_RAL9005',
    'Product_handle_Aluminum (AL)', 'Product_handle_Plastic (PL)',
    'Product_handle_Steel (ST)', 'Product_hinge_Left',
    'Product_hinge_Right', 'Product_packing_Cardbox',
    'Product_packing_Wooden box'
]

batch_sizes = [1, 10, 20, 30, 50, 100]

st.title("⚙️ AI-Powered Product Cost Predictor")
st.write(
    "Leverage intelligent modeling to forecast costs for various production volumes. "
    "Just enter your product specs — we’ll do the math."
)

col1, col2, col3 = st.columns(3)

with col1:
    with st.expander('Product Type'):
        Type = st.selectbox('Type:', ['Select type...'] + type_options)
        Category = st.selectbox('Category:', ['Select category...'] + list(category_names.keys()))
        if Category != 'Select category...':
            Subcategory = st.selectbox('Subcategory:', ['Select subcategory...'] + category_names[Category])
        else:
            Subcategory = st.selectbox('Subcategory:', ['Select category first...'])

with col2:
    with st.expander('Product Attributes'):
        Color = st.selectbox('Color:', ['Select color...'] + color_options)
        Hinge = st.selectbox('Hinge:', ['Select hinge...'] + hinge_options)
        Handle = st.selectbox('Handle:', ['Select handle...'] + handle_options)
        Packing = st.selectbox('Packing:', ['Select packing...'] + packing_options)

with col3:
    with st.expander('Product Dimension'):
        Width = st.slider('Width (mm):', 614, 750, step=10)
        Height = st.slider('Height (mm):', 1200, 1890, step=10)

if st.button("Calculate Cost"):
    required_fields = [Type, Category, Subcategory, Color, Hinge, Handle, Packing]
    if any(field.startswith("Select") for field in required_fields):
        st.warning("Please select all required options before proceeding.")
    else:
        rows = []
        for batch in batch_sizes:
            # Initialize a row with zeros for all required columns
            row = {col: 0 for col in target_columns}

            # Fill in basic numerical and batch size values
            row['Width'] = Width
            row['Height'] = Height
            row['Product_batch'] = batch
            # Product_costs will be predicted later, so keep 0 for now

            # Encode product type (Door or Frame)
            row['Product_type_Door'] = 1 if Type == 'Door' else 0
            row['Product_type_Frame'] = 1 if Type == 'Frame' else 0

            # Encode product category
            for cat in ['Commercial', 'Design', 'Industrial', 'Residential', 'Security']:
                key = f'Product_category_{cat}'
                row[key] = 1 if Category == cat else 0

            # Encode product subcategory
            for subcat in [
                'Access-controlled', 'Balcony', 'Bulletproof', 'Classic', 'Exterior', 'Fire-rated',
                'Glass-panel', 'Interior', 'Minimalist', 'Office', 'Reinforced', 'Retail',
                'Soundproof', 'Thermal-insulated', 'Warehouse'
            ]:
                key = f'Product_subcategory_{subcat}'
                row[key] = 1 if Subcategory == subcat else 0

            # Encode color option
            for col_opt in ['RAL7024', 'RAL7035', 'RAL9005']:
                key = f'Product_color_{col_opt}'
                row[key] = 1 if Color == col_opt else 0

            # Encode handle option
            for handle_opt in ['Aluminum (AL)', 'Plastic (PL)', 'Steel (ST)']:
                key = f'Product_handle_{handle_opt}'
                row[key] = 1 if Handle == handle_opt else 0

            # Encode hinge side
            row['Product_hinge_Left'] = 1 if Hinge == 'Left' else 0
            row['Product_hinge_Right'] = 1 if Hinge == 'Right' else 0

            # Encode packing type
            row['Product_packing_Cardbox'] = 1 if Packing == 'Cardbox' else 0
            row['Product_packing_Wooden box'] = 1 if Packing == 'Wooden box' else 0

            rows.append(row)

        df = pd.DataFrame(rows)

        # Predict product costs using the model
        # The model does not expect the Product_costs column, so drop it before prediction
        X = df.drop(columns=['Product_costs'])
        df['Product_costs'] = model.predict(X)

        # Construct product description string for display
        # Format: type_category_subcategory_color_handle_hinge_widthxheight_packing
        product_desc = f"{Type}_{Category}_{Subcategory}_{Color}_{Handle}_{Hinge}_{Width}x{Height}_{Packing}"

        # Add product description as a new column in DataFrame
        df['product_desc'] = product_desc

        # Round predicted costs to 1 decimal place
        df['Product_costs'] = df['Product_costs'].round(1)

        # Select and rename columns for output display
        result_df = df[['product_desc', 'Product_batch', 'Product_costs']].rename(columns={
            'product_desc': 'Product Description',
            'Product_batch': 'Production Batch',
            'Product_costs': 'Costs [€]'
        })

        # Display the prediction results table
        st.write("### Prediction Results")
        st.dataframe(result_df)
