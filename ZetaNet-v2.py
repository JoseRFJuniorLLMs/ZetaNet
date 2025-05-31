import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
import warnings
warnings.filterwarnings('ignore')

class ImprovedZetaNet(nn.Module):
    def __init__(self, input_size=2, hidden_sizes=[128, 256, 128, 64], output_size=2, dropout_rate=0.1):
        super(ImprovedZetaNet, self).__init__()
        
        # Construir camadas dinamicamente
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Camada de saída sem ativação
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Inicialização Xavier/Glorot
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        return self.network(x)

class ZetaTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            # Gradient clipping para estabilidade
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return total_loss / num_batches
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                predictions = self.model(batch_x)
                loss = criterion(predictions, batch_y)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
    
    def train(self, train_loader, val_loader, epochs=200, learning_rate=0.001, patience=20):
        # Usar Adam com weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
        )
        
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Iniciando treinamento por {epochs} épocas...")
        print("-" * 60)
        
        for epoch in range(epochs):
            # Treinar
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validar
            val_loss = self.validate(val_loader, criterion)
            
            # Atualizar scheduler
            scheduler.step(val_loss)
            
            # Salvar histórico
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Salvar melhor modelo
                torch.save(self.model.state_dict(), 'best_zetanet.pth')
            else:
                patience_counter += 1
            
            # Print progress
            if (epoch + 1) % 20 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Época {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping na época {epoch+1}")
                break
        
        # Carregar melhor modelo
        self.model.load_state_dict(torch.load('best_zetanet.pth'))
        print(f"\nTreinamento concluído! Melhor perda de validação: {best_val_loss:.6f}")

def parse_complex_improved(value):
    """Função melhorada para parsing de números complexos"""
    if pd.isna(value):
        return np.nan
    
    value = str(value).strip()
    
    # Remover parênteses
    value = value.replace('(', '').replace(')', '')
    
    # Substituir vírgulas por pontos
    value = value.replace(',', '.')
    
    # Casos especiais
    if value == '' or value.lower() == 'nan':
        return np.nan
    
    try:
        # Se não tem 'j' ou 'i', adicionar 'j' no final
        if 'j' not in value.lower() and 'i' not in value.lower():
            if '+' in value or '-' in value[1:]:  # Tem parte real e imaginária
                value += 'j'
            else:  # Só parte real
                return complex(float(value), 0)
        
        # Substituir 'i' por 'j'
        value = value.replace('i', 'j')
        
        return complex(value)
    except (ValueError, TypeError):
        return np.nan

def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """Carrega e preprocessa os dados com melhor tratamento de erros"""
    print("Carregando dados...")
    
    try:
        data = pd.read_csv(filepath)
        print(f"Dados carregados: {len(data)} amostras")
    except FileNotFoundError:
        print(f"Arquivo {filepath} não encontrado!")
        return None
    
    # Limpar e converter dados complexos
    print("Processando números complexos...")
    data['s'] = data['s'].apply(parse_complex_improved)
    data['zeta(s)'] = data['zeta(s)'].apply(parse_complex_improved)
    
    # Remover valores inválidos
    initial_len = len(data)
    data = data.dropna()
    final_len = len(data)
    
    if final_len < initial_len:
        print(f"Removidas {initial_len - final_len} amostras inválidas")
    
    if len(data) == 0:
        print("Nenhum dado válido encontrado!")
        return None
    
    # Separar partes real e imaginária
    data['s_real'] = data['s'].apply(lambda x: x.real)
    data['s_imag'] = data['s'].apply(lambda x: x.imag)
    data['zeta_real'] = data['zeta(s)'].apply(lambda x: x.real)
    data['zeta_imag'] = data['zeta(s)'].apply(lambda x: x.imag)
    
    # Preparar features e targets
    X = data[['s_real', 's_imag']].values
    y = data[['zeta_real', 'zeta_imag']].values
    
    # Split treino/validação
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Normalização robusta
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)
    
    # Converter para tensores
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)
    
    print(f"Dados preprocessados:")
    print(f"  Treino: {len(X_train_tensor)} amostras")
    print(f"  Validação: {len(X_val_tensor)} amostras")
    
    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'scalers': (scaler_X, scaler_y),
        'raw_data': data
    }

def create_data_loaders(data_dict, batch_size=64):
    """Cria DataLoaders do PyTorch"""
    train_dataset = torch.utils.data.TensorDataset(
        data_dict['train'][0], data_dict['train'][1]
    )
    val_dataset = torch.utils.data.TensorDataset(
        data_dict['val'][0], data_dict['val'][1]
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader

def plot_results(trainer, data_dict, model):
    """Plota resultados do treinamento e predições"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Curvas de perda
    axes[0,0].plot(trainer.train_losses, label='Treino', alpha=0.8)
    axes[0,0].plot(trainer.val_losses, label='Validação', alpha=0.8)
    axes[0,0].set_xlabel('Época')
    axes[0,0].set_ylabel('MSE Loss')
    axes[0,0].set_title('Curvas de Aprendizado')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    
    # 2. Predições vs Real (parte real)
    model.eval()
    with torch.no_grad():
        X_val, y_val = data_dict['val']
        predictions = model(X_val)
        
        # Denormalizar
        scaler_y = data_dict['scalers'][1]
        y_val_denorm = scaler_y.inverse_transform(y_val.numpy())
        pred_denorm = scaler_y.inverse_transform(predictions.numpy())
    
    axes[0,1].scatter(y_val_denorm[:, 0], pred_denorm[:, 0], alpha=0.6, s=1)
    axes[0,1].plot([y_val_denorm[:, 0].min(), y_val_denorm[:, 0].max()], 
                   [y_val_denorm[:, 0].min(), y_val_denorm[:, 0].max()], 'r--')
    axes[0,1].set_xlabel('ζ(s) Real - Valor Real')
    axes[0,1].set_ylabel('ζ(s) Real - Predição')
    axes[0,1].set_title('Parte Real: Predição vs Real')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Predições vs Real (parte imaginária)
    axes[1,0].scatter(y_val_denorm[:, 1], pred_denorm[:, 1], alpha=0.6, s=1)
    axes[1,0].plot([y_val_denorm[:, 1].min(), y_val_denorm[:, 1].max()], 
                   [y_val_denorm[:, 1].min(), y_val_denorm[:, 1].max()], 'r--')
    axes[1,0].set_xlabel('ζ(s) Imag - Valor Real')
    axes[1,0].set_ylabel('ζ(s) Imag - Predição')
    axes[1,0].set_title('Parte Imaginária: Predição vs Real')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Distribuição dos erros
    errors_real = np.abs(y_val_denorm[:, 0] - pred_denorm[:, 0])
    errors_imag = np.abs(y_val_denorm[:, 1] - pred_denorm[:, 1])
    
    axes[1,1].hist(errors_real, bins=50, alpha=0.7, label='Erro Parte Real')
    axes[1,1].hist(errors_imag, bins=50, alpha=0.7, label='Erro Parte Imag')
    axes[1,1].set_xlabel('Erro Absoluto')
    axes[1,1].set_ylabel('Frequência')
    axes[1,1].set_title('Distribuição dos Erros')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('zetanet_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas
    print(f"\nEstatísticas de Erro:")
    print(f"Erro médio (parte real): {errors_real.mean():.6f}")
    print(f"Erro médio (parte imag): {errors_imag.mean():.6f}")
    print(f"Erro máximo (parte real): {errors_real.max():.6f}")
    print(f"Erro máximo (parte imag): {errors_imag.max():.6f}")

def main():
    start_time = time()
    
    # Configurações
    FILEPATH = "D:/dev/ZetaNet/content/combined_zeta_data.csv" 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")
    
    # Carregar e preprocessar dados
    data_dict = load_and_preprocess_data(FILEPATH)
    if data_dict is None:
        return
    
    # Criar data loaders
    train_loader, val_loader = create_data_loaders(data_dict, batch_size=128)
    
    # Criar modelo melhorado
    model = ImprovedZetaNet(
        input_size=2,
        hidden_sizes=[128, 256, 256, 128, 64],
        output_size=2,
        dropout_rate=0.1
    )
    
    print(f"\nArquitetura do modelo:")
    print(model)
    
    # Contar parâmetros
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParâmetros totais: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")
    
    # Treinar modelo
    trainer = ZetaTrainer(model, DEVICE)
    trainer.train(
        train_loader, val_loader, 
        epochs=300, 
        learning_rate=0.001, 
        patience=30
    )
    
    # Plotar resultados
    plot_results(trainer, data_dict, model)
    
    end_time = time()
    print(f"\nTempo total de execução: {(end_time - start_time):.2f} segundos")

if __name__ == "__main__":
    main()