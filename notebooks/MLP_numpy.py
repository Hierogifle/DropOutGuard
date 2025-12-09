# 📊 MLP COMPLET - GRID SEARCH EXHAUSTIF (MODE COMPLET)
# ════════════════════════════════════════════════════════════════════════════
# Description: MLP avec TOUS les hyperparamètres à tuner - MODE COMPLET
# - Teste TOUTES les combinaisons possibles
# - Charge données FAMD déjà faites
# - Early stopping (ON/OFF)
# - Différentes activations (ReLU, Sigmoid, Tanh)
# - Dropout
# - Optimizers (SGD, Momentum, Adam)
# Date: 2025-12-06
# ════════════════════════════════════════════════════════════════════════════



# ═══════════════════════════════════════════════════════════════════════════════
# 1️⃣  IMPORTATIONS
# ═══════════════════════════════════════════════════════════════════════════════


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')



# ═══════════════════════════════════════════════════════════════════════════════
# 2️⃣  ⚙️  HYPERPARAMÈTRES À TESTER - MODE COMPLET
# ═══════════════════════════════════════════════════════════════════════════════


print("="*80)
print("⚙️  PARAMÈTRES À TUNER - MODE COMPLET")
print("="*80)


# ARCHITECTURE
HIDDEN_SIZES_OPTIONS = [
    [32], [64], [128], [256],              # 1 couche
    [64, 32], [128, 64]      # 2 couches
]

# APPRENTISSAGE
LEARNING_RATE_OPTIONS = [0.001, 0.01, 0.05]

EPOCHS_OPTIONS = [200, 500, 1000]

BATCH_SIZE_OPTIONS = [32]

# RÉGULARISATION
REG_LAMBDA_OPTIONS = [0.0, 0.01]

DROPOUT_OPTIONS = [0.0, 0.3]  # Dropout rate

# FONCTIONS D'ACTIVATION
ACTIVATION_OPTIONS = ['relu', 'tanh']

# EARLY STOPPING
EARLY_STOPPING_OPTIONS = [False, True]

# OPTIMIZER
OPTIMIZER_OPTIONS = ['sgd', 'adam']


# CALCUL DU TOTAL
total = (len(HIDDEN_SIZES_OPTIONS) * len(LEARNING_RATE_OPTIONS) * 
         len(EPOCHS_OPTIONS) * len(BATCH_SIZE_OPTIONS) * len(REG_LAMBDA_OPTIONS) *
         len(DROPOUT_OPTIONS) * len(ACTIVATION_OPTIONS) * len(EARLY_STOPPING_OPTIONS) *
         len(OPTIMIZER_OPTIONS))

print(f"\n✅ Architectures: {len(HIDDEN_SIZES_OPTIONS)}")
print(f"✅ Learning rates: {len(LEARNING_RATE_OPTIONS)}")
print(f"✅ Epochs: {len(EPOCHS_OPTIONS)}")
print(f"✅ Batch sizes: {len(BATCH_SIZE_OPTIONS)}")
print(f"✅ Regularizations: {len(REG_LAMBDA_OPTIONS)}")
print(f"✅ Dropout rates: {len(DROPOUT_OPTIONS)}")
print(f"✅ Activations: {len(ACTIVATION_OPTIONS)}")
print(f"✅ Early Stopping: {len(EARLY_STOPPING_OPTIONS)}")
print(f"✅ Optimizers: {len(OPTIMIZER_OPTIONS)}")
print(f"\n📊 TOTAL DE COMBINAISONS: {total:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3️⃣  CLASSE MLP AVEC TOUTES LES FEATURES
# ═══════════════════════════════════════════════════════════════════════════════


class MLP:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, 
                 reg_lambda=0.0, dropout_rate=0.0, activation='relu', optimizer='sgd'):
        
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        
        # Initialiser poids pour toutes les couches
        self.weights = []
        self.biases = []
        
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2/layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Momentum pour optimizer
        if optimizer == 'momentum':
            self.momentum_w = [np.zeros_like(w) for w in self.weights]
            self.momentum_b = [np.zeros_like(b) for b in self.biases]
            self.momentum_beta = 0.9
        
        # Adam pour optimizer
        if optimizer == 'adam':
            self.m_w = [np.zeros_like(w) for w in self.weights]
            self.v_w = [np.zeros_like(w) for w in self.weights]
            self.m_b = [np.zeros_like(b) for b in self.biases]
            self.v_b = [np.zeros_like(b) for b in self.biases]
            self.t = 0
            self.beta1 = 0.9
            self.beta2 = 0.999
            self.epsilon = 1e-8
    
    
    def _activation(self, x):
        """Fonction d'activation"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
    
    
    def _activation_derivative(self, a, z):
        """Dérivée d'activation"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - a**2
    
    
    def softmax(self, x):
        """Softmax"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    
    def forward(self, X, training=False):
        """Forward pass avec dropout"""
        self.activations = [X]
        self.z_values = []
        self.dropout_masks = []
        
        # Couches cachées
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self._activation(z)
            
            # Dropout
            if training and self.dropout_rate > 0:
                mask = np.random.binomial(1, 1 - self.dropout_rate, a.shape) / (1 - self.dropout_rate)
                a = a * mask
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(None)
            
            self.activations.append(a)
        
        # Couche sortie (softmax)
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.softmax(z)
        self.activations.append(a)
        
        return a
    
    
    def backward(self, X, y):
        """Backward pass"""
        n_samples = X.shape[0]
        y_one_hot = np.eye(self.output_size)[y]
        
        # Gradient sortie
        dz = self.activations[-1] - y_one_hot
        
        d_weights = []
        d_biases = []
        
        # Couche sortie
        d_weights.append(np.dot(self.activations[-2].T, dz) / n_samples)
        d_biases.append(np.sum(dz, axis=0, keepdims=True) / n_samples)
        
        # Couches cachées
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * self._activation_derivative(self.activations[i+1], self.z_values[i])
            
            # Dropout backward
            if self.dropout_masks[i] is not None:
                dz = dz * self.dropout_masks[i]
            
            if i == 0:
                d_weights.insert(0, np.dot(X.T, dz) / n_samples)
            else:
                d_weights.insert(0, np.dot(self.activations[i].T, dz) / n_samples)
            d_biases.insert(0, np.sum(dz, axis=0, keepdims=True) / n_samples)
        
        # Régularisation L2
        for i in range(len(d_weights)):
            d_weights[i] += (self.reg_lambda / n_samples) * self.weights[i]
        
        return d_weights, d_biases
    
    
    def update_weights_sgd(self, d_weights, d_biases):
        """SGD simple"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]
    
    
    def update_weights_momentum(self, d_weights, d_biases):
        """SGD avec momentum"""
        for i in range(len(self.weights)):
            self.momentum_w[i] = self.momentum_beta * self.momentum_w[i] - self.learning_rate * d_weights[i]
            self.momentum_b[i] = self.momentum_beta * self.momentum_b[i] - self.learning_rate * d_biases[i]
            
            self.weights[i] += self.momentum_w[i]
            self.biases[i] += self.momentum_b[i]
    
    
    def update_weights_adam(self, d_weights, d_biases):
        """Adam optimizer"""
        self.t += 1
        
        for i in range(len(self.weights)):
            # Weights
            self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * d_weights[i]
            self.v_w[i] = self.beta2 * self.v_w[i] + (1 - self.beta2) * (d_weights[i]**2)
            
            m_hat_w = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat_w = self.v_w[i] / (1 - self.beta2**self.t)
            
            self.weights[i] -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
            
            # Biases
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * d_biases[i]
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (d_biases[i]**2)
            
            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)
            
            self.biases[i] -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
    
    
    def compute_loss(self, y_pred, y):
        """Cross-entropy loss"""
        n_samples = y_pred.shape[0]
        y_one_hot = np.eye(self.output_size)[y]
        
        loss = -np.mean(np.log(y_pred + 1e-15) * y_one_hot)
        
        reg_loss = (self.reg_lambda / (2 * n_samples)) * sum(np.sum(w**2) for w in self.weights)
        
        return loss + reg_loss
    
    
    def train(self, X_train, y_train, X_val, y_val, epochs=500, batch_size=32, early_stopping=False, patience=20):
        """Entraînement avec early stopping optionnel"""
        train_loss_hist = []
        val_loss_hist = []
        best_val_loss = np.inf
        patience_counter = 0
        epochs_actual = 0
        
        for epoch in range(epochs):
            # Mini-batch
            indices = np.random.permutation(len(X_train))[:batch_size]
            X_batch = X_train[indices]
            y_batch = y_train[indices]
            
            # Forward + Backward
            y_pred = self.forward(X_batch, training=True)
            train_loss = self.compute_loss(y_pred, y_batch)
            train_loss_hist.append(train_loss)
            
            d_weights, d_biases = self.backward(X_batch, y_batch)
            
            # Update selon optimizer
            if self.optimizer == 'sgd':
                self.update_weights_sgd(d_weights, d_biases)
            elif self.optimizer == 'momentum':
                self.update_weights_momentum(d_weights, d_biases)
            elif self.optimizer == 'adam':
                self.update_weights_adam(d_weights, d_biases)
            
            epochs_actual += 1
            
            # Validation loss (early stopping)
            if early_stopping:
                y_pred_val = self.forward(X_val, training=False)
                val_loss = self.compute_loss(y_pred_val, y_val)
                val_loss_hist.append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break
        
        return train_loss_hist, val_loss_hist, epochs_actual
    
    
    def predict(self, X):
        """Prédictions"""
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    
    def evaluate(self, X, y):
        """Évaluation"""
        y_pred = self.forward(X, training=False)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y, y_pred_labels)
        loss = self.compute_loss(y_pred, y)
        return accuracy, loss



# ═══════════════════════════════════════════════════════════════════════════════
# 4️⃣  CHARGEMENT DONNÉES DÉJÀ PRÉPARÉES
# ═══════════════════════════════════════════════════════════════════════════════


print("\n" + "="*80)
print("📊 CHARGEMENT DES DONNÉES (DÉJÀ FAITES)")
print("="*80)

# Charger données déjà faites
df_train = pd.read_csv(Path.cwd().parent / 'data' / 'data_train_encoded.csv', sep=';')
X_train = df_train.drop('Target', axis=1).values  # Tout sauf Target
y_train = df_train['Target'].values               # Que Target

# Charger données déjà faites
df_test = pd.read_csv(Path.cwd().parent / 'data' / 'data_test_encoded.csv', sep=';')
X_test = df_test.drop('Target', axis=1).values  # Tout sauf Target
y_test = df_test['Target'].values               # Que Target

# Split validation (20% du train)
split_idx = int(0.8 * len(X_train))
X_train_split = X_train[:split_idx]
y_train_split = y_train[:split_idx]
X_val = X_train[split_idx:]
y_val = y_train[split_idx:]

print(f"✅ Train: {X_train_split.shape}")
print(f"✅ Val:   {X_val.shape}")
print(f"✅ Test:  {X_test.shape}")
print(f"✅ Classes: {len(np.unique(y_train_split))}")



# ═══════════════════════════════════════════════════════════════════════════════
# 5️⃣  GRID SEARCH COMPLET
# ═══════════════════════════════════════════════════════════════════════════════


print("\n" + "="*80)
print("🔍 GRID SEARCH EXHAUSTIF - EN COURS")
print("="*80)

results = []
current = 1

for hidden_sizes in HIDDEN_SIZES_OPTIONS:
    for lr in LEARNING_RATE_OPTIONS:
        for epochs in EPOCHS_OPTIONS:
            for batch_size in BATCH_SIZE_OPTIONS:
                for reg_lambda in REG_LAMBDA_OPTIONS:
                    for dropout in DROPOUT_OPTIONS:
                        for activation in ACTIVATION_OPTIONS:
                            for early_stopping in EARLY_STOPPING_OPTIONS:
                                for optimizer in OPTIMIZER_OPTIONS:
                                    
                                    # Créer MLP
                                    mlp = MLP(
                                        input_size=X_train_split.shape[1],
                                        hidden_sizes=hidden_sizes,
                                        output_size=len(np.unique(y_train_split)),
                                        learning_rate=lr,
                                        reg_lambda=reg_lambda,
                                        dropout_rate=dropout,
                                        activation=activation,
                                        optimizer=optimizer
                                    )
                                    
                                    # Entraîner
                                    _, _, epochs_actual = mlp.train(X_train_split, y_train_split, X_val, y_val, 
                                                                    epochs=epochs, batch_size=batch_size, 
                                                                    early_stopping=early_stopping, patience=20)
                                    
                                    # Évaluer
                                    train_acc, train_loss = mlp.evaluate(X_train_split, y_train_split)
                                    test_acc, test_loss = mlp.evaluate(X_test, y_test)
                                    
                                    results.append({
                                        'hidden_sizes': str(hidden_sizes),
                                        'lr': lr,
                                        'epochs': epochs,
                                        'batch_size': batch_size,
                                        'reg_lambda': reg_lambda,
                                        'dropout': dropout,
                                        'activation': activation,
                                        'early_stopping': early_stopping,
                                        'optimizer': optimizer,
                                        'epochs_actual': epochs_actual,
                                        'train_acc': train_acc,
                                        'train_loss': train_loss,
                                        'test_acc': test_acc,
                                        'test_loss': test_loss
                                    })
                                    
                                    overfit = train_acc - test_acc
                                    
                                    print(f"[{current:,}/{total:,}] {str(hidden_sizes):20s} | "
                                          f"{activation:6s} | LR:{lr:.5f} | Drop:{dropout:.1f} | "
                                          f"ES:{str(early_stopping):5s} | Opt:{optimizer:8s} | "
                                          f"Train:{train_acc:.2%} | Test:{test_acc:.2%} | "
                                          f"Overfit:{overfit:+.2%}")
                                    
                                    current += 1

# Sauvegarder résultats
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_acc', ascending=False)

graphique_folder = Path.cwd() / 'graphiques'
graphique_folder.mkdir(exist_ok=True)

results_df.to_csv(graphique_folder / 'mlp_grid_search_results_complete.csv', index=False)

print("\n" + "="*80)
print("🏆 TOP 15 MEILLEURES COMBINAISONS")
print("="*80)
print(results_df.head(15).to_string(index=False))

print("\n" + "="*80)
print("⚠️  TOP 15 PIRES COMBINAISONS (OVERFITTING)")
print("="*80)
results_df['overfit'] = results_df['train_acc'] - results_df['test_acc']
print(results_df.nlargest(15, 'overfit')[['hidden_sizes', 'activation', 'lr', 'dropout', 'optimizer', 'train_acc', 'test_acc', 'overfit']].to_string(index=False))

# Meilleurs params
best = results_df.iloc[0]
print(f"\n" + "="*80)
print(f"✅ MEILLEURE COMBINAISON TROUVÉE")
print("="*80)
for col in ['hidden_sizes', 'lr', 'epochs', 'batch_size', 'reg_lambda', 'dropout', 'activation', 'early_stopping', 'optimizer', 'epochs_actual', 'train_acc', 'test_acc']:
    print(f"  {col:20s}: {best[col]}")

print("\n" + "="*80)
print("📊 STATISTIQUES")
print("="*80)
print(f"✅ Meilleure accuracy test:   {results_df['test_acc'].max():.2%}")
print(f"✅ Pire accuracy test:        {results_df['test_acc'].min():.2%}")
print(f"✅ Moyenne accuracy test:     {results_df['test_acc'].mean():.2%}")
print(f"✅ Meilleur overfitting:      {results_df['overfit'].min():+.2%}")
print(f"✅ Pire overfitting:          {results_df['overfit'].max():+.2%}")

print(f"\n✅ Résultats sauvegardés dans: graphiques/mlp_grid_search_results_complete.csv")
print("="*80)