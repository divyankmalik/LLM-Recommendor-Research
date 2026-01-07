# -*- coding: utf-8 -*-
"""
HOT (Hypergraph-of-Thought) Recommendation System
Complete Implementation for Google Colab
"""

# Install Dependencies
print("Installing required packages...")


!pip install -q torch pandas numpy scikit-learn scipy openpyxl matplotlib seaborn

print("All packages installed successfully!\n")

# Import Libraries
import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
import torch.nn.functional as F
from datetime import timedelta
import json
import re
from scipy.sparse import csr_matrix
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt
import seaborn as sns

print("All libraries imported successfully!")

# Data Loading & Preprocessing Functions
def load_amazon_reviews(filepath: str) -> List[Dict]:
    """Load Amazon Reviews dataset from CSV or Excel"""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx') or filepath.endswith('.xls'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file format. Use .csv, .xlsx, or .xls")

    df_mapped = pd.DataFrame({
        'user_id': df['review_hash_id'],
        'item_id': df['upc'],
        'rating': df['review_rating'],
        'timestamp': df['review_date'],
        'category': df['category']
    })

    reviews = df_mapped.to_dict('records')
    return reviews

def create_demo_data(n_users: int = 50, n_items: int = 100, n_interactions: int = 500) -> List[Dict]:
    """Create demo dataset for testing"""
    demo_data = []
    users = [f"user_{i}" for i in range(n_users)]
    items = [f"item_{i}" for i in range(n_items)]
    categories = ["Electronics", "Books", "Home", "Sports", "Fashion"]

    for i in range(n_interactions):
        demo_data.append({
            'user_id': np.random.choice(users),
            'item_id': np.random.choice(items),
            'rating': np.random.randint(1, 6),
            'timestamp': pd.Timestamp('2024-01-01') + pd.Timedelta(days=i//10),
            'category': np.random.choice(categories)
        })

    return demo_data

def preprocess_interactions(raw_data: List[Dict],
                           min_user_interactions: int = 5,
                           min_item_interactions: int = 5,
                           session_window_minutes: int = 30) -> List[Dict]:
    """Preprocess raw reviews into hyperedge format"""
    df = pd.DataFrame(raw_data)

    # Filter low-frequency users and items
    user_counts = df['user_id'].value_counts()
    valid_users = user_counts[user_counts >= min_user_interactions].index
    df = df[df['user_id'].isin(valid_users)]

    item_counts = df['item_id'].value_counts()
    valid_items = item_counts[item_counts >= min_item_interactions].index
    df = df[df['item_id'].isin(valid_items)]

    # Sort by user and timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['user_id', 'timestamp'])

    # Create sessions (hyperedges)
    hyperedges = []

    for user_id in df['user_id'].unique():
        user_data = df[df['user_id'] == user_id].copy()

        # Group interactions by time windows
        user_data['time_diff'] = user_data['timestamp'].diff()
        user_data['new_session'] = (user_data['time_diff'] > timedelta(minutes=session_window_minutes)) | user_data['time_diff'].isna()
        user_data['session_id'] = user_data['new_session'].cumsum()

        # Create hyperedges for each session
        for session_id, session_data in user_data.groupby('session_id'):
            items = session_data['item_id'].tolist()
            categories = session_data['category'].unique().tolist()
            avg_rating = session_data['rating'].mean()
            timestamp = session_data['timestamp'].iloc[0]

            hyperedge = {
                'user': str(user_id),
                'items': [str(item) for item in items],
                'context': {
                    'categories': categories,
                    'avg_rating': float(avg_rating),
                    'session_length': len(items),
                    'primary_category': categories[0] if categories else 'Unknown'
                },
                'timestamp': timestamp,
                'action': 'review'
            }

            hyperedges.append(hyperedge)

    return hyperedges

def build_incidence_matrix(hyperedges: List[Dict]) -> Tuple[torch.Tensor, Dict, Dict]:
    """Build hypergraph incidence matrix H"""
    # Collect all unique nodes (users + items)
    all_nodes = set()
    for edge in hyperedges:
        all_nodes.add(f"user_{edge['user']}")
        for item in edge['items']:
            all_nodes.add(f"item_{item}")

    node_to_idx = {node: idx for idx, node in enumerate(sorted(all_nodes))}
    edge_to_idx = {i: i for i in range(len(hyperedges))}

    # Build incidence matrix
    n_nodes = len(node_to_idx)
    n_edges = len(hyperedges)
    H = torch.zeros(n_nodes, n_edges)

    for edge_idx, edge in enumerate(hyperedges):
        user_node = f"user_{edge['user']}"
        H[node_to_idx[user_node], edge_idx] = 1

        for item in edge['items']:
            item_node = f"item_{item}"
            H[node_to_idx[item_node], edge_idx] = 1

    return H, node_to_idx, edge_to_idx

print(" Data preprocessing functions loaded!")

# Hypergraph Neural Network
class HypergraphConv(nn.Module):
    """Hypergraph convolution layer"""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        # Compute degree matrices
        D_v = torch.sum(H, dim=1)
        D_e = torch.sum(H, dim=0)

        # Add epsilon to avoid division by zero
        inv_sqrt_D_v = torch.pow(D_v + 1e-10, -0.5)
        inv_D_e = torch.pow(D_e + 1e-10, -1.0)

        # Apply transformation
        X = self.W(X)

        # Hypergraph convolution
        X = inv_sqrt_D_v.unsqueeze(1) * X
        X = torch.matmul(H, inv_D_e.unsqueeze(1) * torch.matmul(H.t(), X))
        X = inv_sqrt_D_v.unsqueeze(1) * X

        return X

class BaselineHypergraphModel(nn.Module):
    """Baseline hypergraph recommendation model"""
    def __init__(self, n_nodes: int, embed_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.embed_dim = embed_dim

        # Node embeddings
        self.node_embeddings = nn.Embedding(n_nodes, embed_dim)
        nn.init.xavier_normal_(self.node_embeddings.weight)

        # Hypergraph convolution layers
        self.conv_layers = nn.ModuleList([
            HypergraphConv(embed_dim, embed_dim) for _ in range(n_layers)
        ])

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        X = self.node_embeddings.weight

        for conv in self.conv_layers:
            X = conv(X, H)
            X = self.activation(X)
            X = self.dropout(X)

        return X

    def predict_score(self, user_idx: int, item_idx: int, embeddings: torch.Tensor) -> torch.Tensor:
        user_emb = embeddings[user_idx]
        item_emb = embeddings[item_idx]
        score = torch.sum(user_emb * item_emb)
        return score

def train_baseline(model, hyperedges, H, node_to_idx, epochs: int = 50, lr: float = 0.001):
    """Train baseline model with BPR loss"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Prepare user-item pairs
    user_items = defaultdict(list)
    all_items = set()

    for edge in hyperedges:
        user = f"user_{edge['user']}"
        if user in node_to_idx:
            user_idx = node_to_idx[user]
            for item in edge['items']:
                item_node = f"item_{item}"
                if item_node in node_to_idx:
                    item_idx = node_to_idx[item_node]
                    user_items[user_idx].append(item_idx)
                    all_items.add(item_idx)

    all_items = list(all_items)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for user_idx, pos_items in user_items.items():
            for pos_item in pos_items[:5]:
                embeddings = model(H)

                neg_item = np.random.choice(all_items)
                while neg_item in pos_items:
                    neg_item = np.random.choice(all_items)

                pos_score = model.predict_score(user_idx, pos_item, embeddings)
                neg_score = model.predict_score(user_idx, neg_item, embeddings)

                loss = -torch.log(torch.sigmoid(pos_score - neg_score) + 1e-10)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches if n_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_baseline(model, test_hyperedges, H, node_to_idx, k: int = 10) -> Dict[str, float]:
    """Evaluate model with Recall@K and NDCG@K"""
    model.eval()

    with torch.no_grad():
        embeddings = model(H)

    user_items = defaultdict(list)
    all_items = set()

    for edge in test_hyperedges:
        user = f"user_{edge['user']}"
        if user in node_to_idx:
            user_idx = node_to_idx[user]
            for item in edge['items']:
                item_node = f"item_{item}"
                if item_node in node_to_idx:
                    item_idx = node_to_idx[item_node]
                    user_items[user_idx].append(item_idx)
                    all_items.add(item_idx)

    all_items = list(all_items)

    recalls = []
    ndcgs = []

    for user_idx, true_items in user_items.items():
        if len(true_items) == 0:
            continue

        scores = []
        for item_idx in all_items:
            score = model.predict_score(user_idx, item_idx, embeddings)
            scores.append((item_idx, score.item()))

        scores.sort(key=lambda x: x[1], reverse=True)
        top_k_items = [item for item, _ in scores[:k]]

        hits = len(set(top_k_items) & set(true_items))
        recall = hits / min(len(true_items), k)
        recalls.append(recall)

        relevance = [1 if item in true_items else 0 for item in top_k_items]
        if sum(relevance) > 0:
            ideal_relevance = [1] * min(len(true_items), k) + [0] * (k - min(len(true_items), k))
            ndcg = ndcg_score([ideal_relevance], [relevance])
            ndcgs.append(ndcg)

    return {
        'recall': np.mean(recalls) if recalls else 0,
        'ndcg': np.mean(ndcgs) if ndcgs else 0
    }

print("Hypergraph neural network loaded!")

# HOT Reasoning with LLM
@dataclass
class ReasoningPath:
    """Structured reasoning output from LLM"""
    hyperedge_ids: List[int]
    steps: List[str]
    recommendations: List[str]
    confidences: List[float]

class HOTReasoner:
    """LLM-powered reasoning over hypergraph structure"""

    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.hyperedge_history = {}

    def store_user_history(self, user_id: str, hyperedges: List[Dict]):
        self.hyperedge_history[user_id] = hyperedges

    def hyperedge_to_text(self, edge_data: Dict, edge_id: int) -> str:
        user = edge_data['user']
        items = edge_data['items']
        context = edge_data['context']

        if len(items) == 1:
            items_str = items[0]
        elif len(items) == 2:
            items_str = f"{items[0]} and {items[1]}"
        else:
            items_str = ", ".join(items[:-1]) + f", and {items[-1]}"

        description = f"Session {edge_id}: User {user} interacted with {items_str}"

        if 'categories' in context and context['categories']:
            categories = context['categories']
            if len(categories) == 1:
                description += f" in the {categories[0]} category"
            else:
                description += f" across {', '.join(categories)} categories"

        if 'avg_rating' in context:
            description += f" (avg rating: {context['avg_rating']:.1f}/5)"

        return description

    def build_reasoning_prompt(self, user_id: str, user_history: List[str],
                               candidates: List[str]) -> str:
        prompt = f"""You are an expert recommendation system. Analyze the user's interaction history and recommend the best items.

USER ID: {user_id}

INTERACTION HISTORY:
{chr(10).join(f"{i+1}. {hist}" for i, hist in enumerate(user_history))}

CANDIDATE ITEMS TO RECOMMEND:
{', '.join(candidates)}

TASK: Analyze the user's behavior patterns and recommend the top 3 items from the candidates.

Please respond in this EXACT format:

REASONING:
STEP 1: [Analyze what categories/types of items the user prefers]
STEP 2: [Identify key patterns in their interaction history]
STEP 3: [Match candidates to user preferences with specific reasoning]

RECOMMENDATIONS:
1. [Item Name] - Confidence: [0.0-1.0] - Reason: [specific reason based on history]
2. [Item Name] - Confidence: [0.0-1.0] - Reason: [specific reason based on history]
3. [Item Name] - Confidence: [0.0-1.0] - Reason: [specific reason based on history]
"""
        return prompt

    def call_llm(self, prompt: str) -> str:
        if self.use_mock:
            return self._mock_llm_response(prompt)
        else:
            return self._call_claude_api(prompt)

    def _mock_llm_response(self, prompt: str) -> str:
        candidates_section = prompt.split("CANDIDATE ITEMS TO RECOMMEND:")[1].split("TASK:")[0].strip()
        candidates = [c.strip() for c in candidates_section.split(',')]

        response = """REASONING:
STEP 1: The user shows strong preference for electronics and gaming accessories based on their interaction history
STEP 2: They consistently rate items highly (4.0+ ratings) and tend to purchase complementary products in the same session
STEP 3: Among the candidates, items matching their past category preferences should be prioritized

RECOMMENDATIONS:
1. """ + candidates[0] + """ - Confidence: 0.85 - Reason: Aligns with user's preference for similar category items and high ratings
2. """ + (candidates[1] if len(candidates) > 1 else candidates[0]) + """ - Confidence: 0.72 - Reason: Complementary to previous purchases in their history
3. """ + (candidates[2] if len(candidates) > 2 else candidates[0]) + """ - Confidence: 0.68 - Reason: Falls within frequently browsed categories"""

        return response

    def _call_claude_api(self, prompt: str) -> str:
        try:
            import anthropic
            import os
            from google.colab import userdata

            # Try to get API key from Colab secrets
            try:
                api_key = userdata.get('ANTHROPIC_API_KEY')
            except:
                api_key = os.environ.get("ANTHROPIC_API_KEY")

            if not api_key:
                print("No API key found. Using mock response.")
                return self._mock_llm_response(prompt)

            client = anthropic.Anthropic(api_key=api_key)

            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )

            return message.content[0].text
        except Exception as e:
            print(f"Error calling Claude API: {e}")
            print("Falling back to mock response...")
            return self._mock_llm_response(prompt)

    def parse_reasoning(self, llm_output: str) -> ReasoningPath:
        steps = []
        recommendations = []
        confidences = []

        reasoning_section = re.search(r'REASONING:(.*?)RECOMMENDATIONS:', llm_output, re.DOTALL)
        if reasoning_section:
            step_matches = re.findall(r'STEP \d+: (.+)', reasoning_section.group(1))
            steps = [step.strip() for step in step_matches]

        rec_section = re.search(r'RECOMMENDATIONS:(.*?)$', llm_output, re.DOTALL)
        if rec_section:
            rec_lines = rec_section.group(1).strip().split('\n')
            for line in rec_lines:
                if line.strip() and line.strip()[0].isdigit():
                    parts = line.split(' - ')
                    if len(parts) >= 2:
                        item = parts[0].split('.', 1)[1].strip()
                        recommendations.append(item)

                        conf_match = re.search(r'Confidence:\s*([\d.]+)', line)
                        if conf_match:
                            confidences.append(float(conf_match.group(1)))
                        else:
                            confidences.append(0.5)

        return ReasoningPath(
            hyperedge_ids=[],
            steps=steps,
            recommendations=recommendations,
            confidences=confidences
        )

    def reason_and_recommend(self, user_id: str, candidate_items: List[str],
                            top_k: int = 3) -> ReasoningPath:
        if user_id not in self.hyperedge_history:
            print(f"Warning: No history found for user {user_id}")
            return ReasoningPath([], [], [], [])

        user_edges = self.hyperedge_history[user_id]

        history_texts = [
            self.hyperedge_to_text(edge, idx)
            for idx, edge in enumerate(user_edges)
        ]

        prompt = self.build_reasoning_prompt(user_id, history_texts, candidate_items)
        llm_output = self.call_llm(prompt)
        reasoning_path = self.parse_reasoning(llm_output)

        reasoning_path.hyperedge_ids = list(range(len(user_edges)))

        return reasoning_path

print(" HOT reasoner loaded!")

# Main Pipeline
def main():
    """Complete workflow for HOT recommendation system"""

    print("="*70)
    print(" HOT (Hypergraph-of-Thought) Recommendation System")
    print("="*70)

    # Phase 1: Data Loading
    print("\n[Phase 1/4] Loading data...")
    try:
        raw_data = load_amazon_reviews("amazon_reviews.csv")
        print(f"✓ Loaded data from uploaded file")
    except:
        print("No uploaded file found. Creating demo data...")
        raw_data = create_demo_data()
        print(f"✓ Created demo data")

    print(f"  Total interactions: {len(raw_data)}")

    # Preprocess
    print("\nPreprocessing into hyperedges...")
    hyperedges = preprocess_interactions(raw_data, min_user_interactions=3, min_item_interactions=2)
    print(f"✓ Created {len(hyperedges)} hyperedges")

    # Build hypergraph
    print("\nBuilding hypergraph structure...")
    H, node_to_idx, edge_to_idx = build_incidence_matrix(hyperedges)
    print(f"✓ Hypergraph: {H.shape[0]} nodes, {H.shape[1]} edges")

    # Split data
    split_idx = int(0.8 * len(hyperedges))
    train_edges = hyperedges[:split_idx]
    test_edges = hyperedges[split_idx:]
    print(f"✓ Train: {len(train_edges)}, Test: {len(test_edges)}")

    # Phase 2: Train Baseline
    print(f"\n[Phase 2/4] Training Baseline Model...")
    print("This may take 1-2 minutes...")
    baseline = BaselineHypergraphModel(n_nodes=H.shape[0], embed_dim=64, n_layers=2)
    train_baseline(baseline, train_edges, H, node_to_idx, epochs=30, lr=0.001)
    print("✓ Training complete!")

    # Phase 3: Initialize HOT
    print(f"\n[Phase 3/4] Initializing HOT Reasoner...")
    hot = HOTReasoner(use_mock=True)
    print("✓ HOT reasoner ready!")

    # Phase 4: Evaluation
    print(f"\n[Phase 4/4] Running Evaluation...")

    # Evaluate baseline
    baseline_metrics = evaluate_baseline(baseline, test_edges, H, node_to_idx, k=10)
    print(f"✓ Baseline evaluated")

    # Evaluate HOT
    user_histories = defaultdict(list)
    for edge in test_edges:
        user_histories[edge['user']].append(edge)

    hot_recalls = []
    for user_id, edges in list(user_histories.items())[:min(20, len(user_histories))]:
        if len(edges) < 2:
            continue

        hot.store_user_history(user_id, edges[:-1])
        true_items = edges[-1]['items']

        all_items = set()
        for edge in test_edges[:50]:
            all_items.update(edge['items'])
        candidate_items = list(all_items)[:15]

        reasoning = hot.reason_and_recommend(user_id, candidate_items, top_k=10)

        if reasoning.recommendations:
            hits = len(set(reasoning.recommendations[:10]) & set(true_items))
            recall = hits / min(len(true_items), 10)
            hot_recalls.append(recall)

    hot_recall = np.mean(hot_recalls) if hot_recalls else 0
    print(f"✓ HOT evaluated")

    # Display results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"\n🔹 Baseline Model:")
    print(f"  • Recall@10: {baseline_metrics['recall']:.4f}")
    print(f"  • NDCG@10: {baseline_metrics['ndcg']:.4f}")
    print(f"\n🔹 HOT Model:")
    print(f"  • Recall@10: {hot_recall:.4f}")

    improvement = ((hot_recall / baseline_metrics['recall']) - 1) * 100 if baseline_metrics['recall'] > 0 else 0
    print(f"\n HOT Improvement: {improvement:+.1f}%")

    # Demo reasoning
    print("\n" + "="*70)
    print("SAMPLE HOT REASONING")
    print("="*70)

    if test_edges:
        sample_user = test_edges[0]['user']
        sample_items = ["laptop", "mouse", "keyboard", "monitor", "headset"]

        user_edges = [e for e in train_edges if e['user'] == sample_user][:5]
        if user_edges:
            hot.store_user_history(sample_user, user_edges)
            reasoning = hot.reason_and_recommend(sample_user, sample_items, top_k=3)

            print(f"\n👤 User: {sample_user}")
            print(f"\nReasoning Steps:")
            for i, step in enumerate(reasoning.steps, 1):
                print(f"  {i}. {step}")

            print(f"\nRecommendations:")
            for i, (item, conf) in enumerate(zip(reasoning.recommendations, reasoning.confidences), 1):
                print(f"  {i}. {item} (confidence: {conf:.2f})")

    print("\n" + "="*70)
    print("Pipeline Complete!")
    print("="*70)

    # Create visualization
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Model comparison
    models = ['Baseline', 'HOT']
    recalls = [baseline_metrics['recall'], hot_recall]
    ax[0].bar(models, recalls, color=['#3498db', '#e74c3c'])
    ax[0].set_ylabel('Recall@10')
    ax[0].set_title('Model Performance Comparison')
    ax[0].set_ylim(0, max(recalls) * 1.2)
    for i, v in enumerate(recalls):
        ax[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # Plot 2: Metrics comparison
    metrics = ['Recall@10', 'NDCG@10']
    baseline_vals = [baseline_metrics['recall'], baseline_metrics['ndcg']]
    x = np.arange(len(metrics))
    width = 0.35
    ax[1].bar(x, baseline_vals, width, label='Baseline', color='#3498db')
    ax[1].set_ylabel('Score')
    ax[1].set_title('Baseline Metrics')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(metrics)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('hot_results.png', dpi=150, bbox_inches='tight')
    print("\n Visualization saved as 'hot_results.png'")
    plt.show()

    return {
        'baseline': baseline_metrics,
        'hot': {'recall': hot_recall},
        'hyperedges': hyperedges
    }

# Run the complete pipeline
print("\n Starting HOT Recommendation System...\n")
results = main()

print("\n" + "="*70)
print("All done! HOT recommendation system is working!")
print("="*70)
