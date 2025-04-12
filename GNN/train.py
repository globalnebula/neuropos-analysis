from torch_geometric.loader import DataLoader
from drug_gnn import DrugSideEffectGNN
from data_loader import load_drug_side_effect_graph
from sklearn.metrics import classification_report

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data, side_effect_classes = load_drug_side_effect_graph("drugs.csv", "side_effects.csv", "interactions.csv")
data = data.to(device)

model = DrugSideEffectGNN(input_dim=data.num_node_features, hidden_dim=64, output_dim=len(side_effect_classes)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.binary_cross_entropy(out, data.y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

# Evaluation
model.eval()
pred = (model(data.x, data.edge_index).detach().cpu().numpy() > 0.5).astype(int)
true = data.y.cpu().numpy()
print(classification_report(true, pred, target_names=side_effect_classes))
