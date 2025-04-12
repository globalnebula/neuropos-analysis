import torch
from torch_geometric.data import Data
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def load_drug_side_effect_graph(drugs_file, side_effects_file, interactions_file):
    drug_df = pd.read_csv(drugs_file)  # contains drug_id, features...
    se_df = pd.read_csv(side_effects_file)  # drug_id, [side_effect_1, ..., side_effect_n]
    inter_df = pd.read_csv(interactions_file)  # drug_id_1, drug_id_2

    # Node features
    features = torch.tensor(drug_df.drop(columns=['drug_id']).values, dtype=torch.float)
    
    # Edge index (drug-drug interactions)
    edge_index = torch.tensor(inter_df[['drug_id_1', 'drug_id_2']].values.T, dtype=torch.long)

    # Labels (multi-label side effects)
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(se_df['side_effects'].apply(eval))
    labels = torch.tensor(label_matrix, dtype=torch.float)

    return Data(x=features, edge_index=edge_index, y=labels), mlb.classes_
