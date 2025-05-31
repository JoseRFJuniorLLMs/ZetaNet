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
import pickle  # Importado para salvar/carregar scalers e modelo

warnings.filterwarnings('ignore')


# ### CLASSE DO MODELO NEURAL ###
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


# ### CLASSE DO TREINADOR ###
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
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()
        best_val_loss = float('inf')
        patience_counter = 0

        print(f"Iniciando treinamento por {epochs} épocas...")
        print("-" * 60)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss = self.validate(val_loader, criterion)
            scheduler.step(val_loss)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_zetanet.pth')
                print("   -> Melhor modelo salvo!")
            else:
                patience_counter += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Época {epoch + 1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"LR: {current_lr:.2e}")

            if patience_counter >= patience:
                print(f"\nEarly stopping na época {epoch + 1}")
                break

        self.model.load_state_dict(torch.load('best_zetanet.pth'))
        print(f"\nTreinamento concluído! Melhor perda de validação: {best_val_loss:.6f}")


# ### FUNÇÕES DE PRÉ-PROCESSAMENTO E UTILIDADES ###
def parse_complex_improved(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().replace('(', '').replace(')', '').replace(',', '.')
    if value == '' or value.lower() == 'nan':
        return np.nan
    try:
        if 'j' not in value.lower() and 'i' not in value.lower():
            if '+' in value or '-' in value[1:]:
                value += 'j'
            else:
                return complex(float(value), 0)
        value = value.replace('i', 'j')
        return complex(value)
    except (ValueError, TypeError):
        return np.nan


def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    print("Carregando dados...")
    try:
        data = pd.read_csv(filepath)
        print(f"Dados carregados: {len(data)} amostras")
    except FileNotFoundError:
        print(f"Arquivo {filepath} não encontrado!")
        return None

    print("Processando números complexos...")
    data['s'] = data['s'].apply(parse_complex_improved)
    data['zeta(s)'] = data['zeta(s)'].apply(parse_complex_improved)

    initial_len = len(data)
    data = data.dropna()
    final_len = len(data)

    if final_len < initial_len:
        print(f"Removidas {initial_len - final_len} amostras inválidas")
    if len(data) == 0:
        print("Nenhum dado válido encontrado!")
        return None

    data['s_real'] = data['s'].apply(lambda x: x.real)
    data['s_imag'] = data['s'].apply(lambda x: x.imag)
    data['zeta_real'] = data['zeta(s)'].apply(lambda x: x.real)
    data['zeta_imag'] = data['zeta(s)'].apply(lambda x: x.imag)

    X = data[['s_real', 's_imag']].values
    y = data[['zeta_real', 'zeta_imag']].values

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    y_train_scaled = scaler_y.fit_transform(y_train)
    y_val_scaled = scaler_y.transform(y_val)

    X_train_tensor = torch.FloatTensor(X_train_scaled)
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_train_tensor = torch.FloatTensor(y_train_scaled)
    y_val_tensor = torch.FloatTensor(y_val_scaled)

    print(f"Dados preprocessados:")
    print(f"   Treino: {len(X_train_tensor)} amostras")
    print(f"   Validação: {len(X_val_tensor)} amostras")

    return {
        'train': (X_train_tensor, y_train_tensor),
        'val': (X_val_tensor, y_val_tensor),
        'scalers': (scaler_X, scaler_y),
        'raw_data': data
    }


def create_data_loaders(data_dict, batch_size=64):
    train_dataset = torch.utils.data.TensorDataset(data_dict['train'][0], data_dict['train'][1])
    val_dataset = torch.utils.data.TensorDataset(data_dict['val'][0], data_dict['val'][1])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def plot_results(trainer, data_dict, model):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    axes[0, 0].plot(trainer.train_losses, label='Treino', alpha=0.8)
    axes[0, 0].plot(trainer.val_losses, label='Validação', alpha=0.8)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('MSE Loss')
    axes[0, 0].set_title('Curvas de Aprendizado')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')

    model.eval()
    with torch.no_grad():
        X_val, y_val = data_dict['val']
        predictions = model(
            X_val.to(next(model.parameters()).device))  # Garantir que X_val está no mesmo device que o modelo

        scaler_y = data_dict['scalers'][1]
        y_val_denorm = scaler_y.inverse_transform(y_val.cpu().numpy())
        pred_denorm = scaler_y.inverse_transform(predictions.cpu().numpy())

    axes[0, 1].scatter(y_val_denorm[:, 0], pred_denorm[:, 0], alpha=0.6, s=1)
    axes[0, 1].plot([y_val_denorm[:, 0].min(), y_val_denorm[:, 0].max()],
                    [y_val_denorm[:, 0].min(), y_val_denorm[:, 0].max()], 'r--')
    axes[0, 1].set_xlabel('ζ(s) Real - Valor Real')
    axes[0, 1].set_ylabel('ζ(s) Real - Predição')
    axes[0, 1].set_title('Parte Real: Predição vs Real')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].scatter(y_val_denorm[:, 1], pred_denorm[:, 1], alpha=0.6, s=1)
    axes[1, 0].plot([y_val_denorm[:, 1].min(), y_val_denorm[:, 1].max()],
                    [y_val_denorm[:, 1].min(), y_val_denorm[:, 1].max()], 'r--')
    axes[1, 0].set_xlabel('ζ(s) Imag - Valor Real')
    axes[1, 0].set_ylabel('ζ(s) Imag - Predição')
    axes[1, 0].set_title('Parte Imaginária: Predição vs Real')
    axes[1, 0].grid(True, alpha=0.3)

    errors_real = np.abs(y_val_denorm[:, 0] - pred_denorm[:, 0])
    errors_imag = np.abs(y_val_denorm[:, 1] - pred_denorm[:, 1])

    axes[1, 1].hist(errors_real, bins=50, alpha=0.7, label='Erro Parte Real')
    axes[1, 1].hist(errors_imag, bins=50, alpha=0.7, label='Erro Parte Imag')
    axes[1, 1].set_xlabel('Erro Absoluto')
    axes[1, 1].set_ylabel('Frequência')
    axes[1, 1].set_title('Distribuição dos Erros')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_yscale('log')

    plt.tight_layout()
    plt.savefig('zetanet_results_training.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nEstatísticas de Erro:")
    print(f"Erro médio (parte real): {errors_real.mean():.6f}")
    print(f"Erro médio (parte imag): {errors_imag.mean():.6f}")
    print(f"Erro máximo (parte real): {errors_real.max():.6f}")
    print(f"Erro máximo (parte imag): {errors_imag.max():.6f}")


# ### FUNÇÃO PRINCIPAL DE TREINAMENTO ###
def main_train():
    start_time = time()

    # ### MODIFICAÇÃO: Caminho do arquivo ajustado para Windows e evitar SyntaxWarning
    # ### AJUSTE ESTE CAMINHO SE O SEU ARQUIVO .csv ESTIVER EM OUTRO LUGAR ###
    FILEPATH = r"combined_zeta_data.csv"  # Usando 'r' para raw string.
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")

    data_dict = load_and_preprocess_data(FILEPATH)
    if data_dict is None:
        return

    try:
        with open('scaler_X.pkl', 'wb') as f:
            pickle.dump(data_dict['scalers'][0], f)
        with open('scaler_y.pkl', 'wb') as f:
            pickle.dump(data_dict['scalers'][1], f)
        print("Scalers scaler_X.pkl e scaler_y.pkl salvos.")
    except Exception as e:
        print(f"Erro ao salvar scalers: {e}")

    train_loader, val_loader = create_data_loaders(data_dict, batch_size=128)

    # ### ATENÇÃO: AJUSTE A ARQUITETURA DO MODELO CONFORME DESEJADO PARA O TREINAMENTO ###
    model_architecture_params_train = {
        'input_size': 2,
        'hidden_sizes': [128, 256, 256, 128, 64],  # Exemplo de arquitetura
        'output_size': 2,
        'dropout_rate': 0.1
    }
    model = ImprovedZetaNet(**model_architecture_params_train).to(DEVICE)

    print(f"\nArquitetura do modelo para TREINAMENTO:")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParâmetros totais: {total_params:,}")
    print(f"Parâmetros treináveis: {trainable_params:,}")

    trainer = ZetaTrainer(model, DEVICE)
    trainer.train(
        train_loader, val_loader,
        epochs=300,  # Ajuste o número de épocas
        learning_rate=0.001,  # Ajuste a taxa de aprendizado
        patience=30  # Ajuste a paciência para early stopping
    )

    plot_results(trainer, data_dict, model)

    print("\n--- Exemplo de Inferência Pós-Treinamento ---")
    try:
        loaded_model_inference = ImprovedZetaNet(**model_architecture_params_train).to(
            DEVICE)  # Usa a mesma arquitetura do treino
        loaded_model_inference.load_state_dict(torch.load('best_zetanet.pth', map_location=DEVICE))
        loaded_model_inference.eval()
        print(f"Modelo 'best_zetanet.pth' carregado com sucesso para inferência.")

        with open('scaler_X.pkl', 'rb') as f:
            loaded_scaler_X_inference = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            loaded_scaler_y_inference = pickle.load(f)
        print(f"Scalers 'scaler_X.pkl' e 'scaler_y.pkl' carregados com sucesso.")

        new_s_real = 0.5
        new_s_imag = 100.0
        new_data_unscaled = np.array([[new_s_real, new_s_imag]])
        new_data_scaled = loaded_scaler_X_inference.transform(new_data_unscaled)
        new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            prediction_scaled = loaded_model_inference(new_data_tensor)
        prediction_unscaled = loaded_scaler_y_inference.inverse_transform(prediction_scaled.cpu().numpy())

        print(f"\n--- Previsão para s = {new_s_real} + {new_s_imag}i ---")
        print(f"ζ(s) previsto: Real={prediction_unscaled[0][0]:.8f}, Imaginária={prediction_unscaled[0][1]:.8f}")

    except FileNotFoundError as e:
        print(f"Erro ao carregar modelo ou scalers para inferência pós-treinamento: {e}.")
    except Exception as e:
        print(f"Ocorreu um erro durante a inferência pós-treinamento: {e}")

    end_time = time()
    print(f"\nTempo total de execução do treinamento: {(end_time - start_time):.2f} segundos")


# --- FUNÇÕES PARA BUSCA DE ZEROS ---
def find_zeros_on_critical_line(model, scaler_X, scaler_y, device, t_min, t_max, num_points=10000, threshold=0.1):
    model.eval()
    s_real = 0.5
    potential_zeros_t = []
    t_values = np.linspace(t_min, t_max, num_points)
    predicted_zeta_magnitudes = []

    print(
        f"\nProcurando zeros na linha crítica para t em [{t_min}, {t_max}] com {num_points} pontos e threshold {threshold}...")

    for i, t_imag in enumerate(t_values):
        s_input_unscaled = np.array([[s_real, t_imag]])
        s_input_scaled = scaler_X.transform(s_input_unscaled)
        s_input_tensor = torch.tensor(s_input_scaled, dtype=torch.float32).to(device)

        with torch.no_grad():
            prediction_scaled = model(s_input_tensor)
        prediction_unscaled = scaler_y.inverse_transform(prediction_scaled.cpu().numpy())

        zeta_real = prediction_unscaled[0, 0]
        zeta_imag = prediction_unscaled[0, 1]
        modulus_zeta = np.sqrt(zeta_real ** 2 + zeta_imag ** 2)
        predicted_zeta_magnitudes.append(modulus_zeta)

        if modulus_zeta < threshold:
            print(
                f"  Zero potencial encontrado em t ≈ {t_imag:.4f} | ζ(0.5 + {t_imag:.4f}i) ≈ {zeta_real:.4f} + {zeta_imag:.4f}i | Módulo: {modulus_zeta:.6f}")
            potential_zeros_t.append(t_imag)

        if (i + 1) % (num_points // 10) == 0:  # Print de progresso
            print(f"  Progresso da busca: {((i + 1) / num_points * 100):.0f}% concluído.")

    print(f"Procura concluída. {len(potential_zeros_t)} zeros potenciais encontrados.")

    plt.figure(figsize=(12, 6))
    plt.plot(t_values, predicted_zeta_magnitudes, label='|ζ(0.5 + it)| (Predito pelo Modelo)', linewidth=0.8)
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Limiar ({threshold})')

    # Para melhor visualização dos zeros encontrados, plotamos eles um pouco acima do eixo x se a escala for log
    y_scatter = threshold / 2 if plt.gca().get_yscale() == 'log' else 0.001
    if potential_zeros_t:
        plt.scatter(potential_zeros_t, [y_scatter] * len(potential_zeros_t), color='green', s=50,
                    label='Zeros Potenciais Encontrados', zorder=5, edgecolor='black')

    plt.xlabel('Parte Imaginária t')
    plt.ylabel('|ζ(0.5 + it)|')
    plt.title('Módulo de ζ(s) Predito na Linha Crítica')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.yscale('log')
    min_magnitude_plot = max(1e-6, np.min(predicted_zeta_magnitudes) / 2) if predicted_zeta_magnitudes else 1e-6
    plt.ylim(bottom=min_magnitude_plot, top=max(predicted_zeta_magnitudes) * 2 if predicted_zeta_magnitudes else 1.0)
    plt.savefig('zetanet_zeros_search.png', dpi=300, bbox_inches='tight')
    plt.show()

    return potential_zeros_t


def main_find_zeros():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {DEVICE}")
    start_time_zeros = time()

    # ### ATENÇÃO: ESTA ARQUITETURA DEVE SER A MESMA USADA PARA TREINAR O MODELO 'best_zetanet.pth' ###
    # Se você treinou com uma arquitetura diferente, altere os parâmetros abaixo correspondentemente.
    model_architecture_params_load = {
        'input_size': 2,
        'hidden_sizes': [128, 256, 256, 128, 64],  # Ajuste se o modelo salvo tiver outra arquitetura
        'output_size': 2,
        'dropout_rate': 0.1  # Ajuste se o modelo salvo tiver outro dropout
    }
    print("\n--- Carregando Modelo e Scalers para Busca de Zeros ---")
    print(f"Tentando carregar modelo com arquitetura: {model_architecture_params_load['hidden_sizes']}")

    try:
        loaded_model_zeros = ImprovedZetaNet(**model_architecture_params_load).to(DEVICE)
        loaded_model_zeros.load_state_dict(torch.load('best_zetanet.pth', map_location=DEVICE))
        loaded_model_zeros.eval()
        print(f"Modelo 'best_zetanet.pth' carregado com sucesso.")

        with open('scaler_X.pkl', 'rb') as f:
            loaded_scaler_X_zeros = pickle.load(f)
        with open('scaler_y.pkl', 'rb') as f:
            loaded_scaler_y_zeros = pickle.load(f)
        print(f"Scalers 'scaler_X.pkl' e 'scaler_y.pkl' carregados com sucesso.")

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado: {e}.")
        print("Certifique-se de que 'best_zetanet.pth', 'scaler_X.pkl' e 'scaler_y.pkl' existem no mesmo diretório.")
        print("Você precisa treinar o modelo primeiro (escolha a opção 'treinar').")
        return
    except RuntimeError as e:
        print(f"Erro ao carregar o estado do modelo (RuntimeError): {e}")
        print("Isso geralmente acontece se a arquitetura do modelo definida aqui (model_architecture_params_load)")
        print("não corresponder exatamente à arquitetura do modelo salvo em 'best_zetanet.pth'.")
        print("Verifique 'hidden_sizes' e outros parâmetros da classe ImprovedZetaNet.")
        return
    except Exception as e:
        print(f"Ocorreu um erro inesperado durante o carregamento: {e}")
        return

    # Parâmetros para a busca de zeros (ajuste conforme necessário)
    t_min_search = 0.0  # Os primeiros zeros conhecidos têm t ≈ 14.1347, 21.0220, 25.0108
    t_max_search = 50.0  # Aumente para procurar mais zeros
    num_points_search = 50000  # Aumente para busca mais fina (mais lento)
    threshold_search = 0.05  # Limiar para |ζ(s)|. Diminua se o modelo for muito preciso, aumente se for menos.

    potential_zeros = find_zeros_on_critical_line(
        loaded_model_zeros,
        loaded_scaler_X_zeros,
        loaded_scaler_y_zeros,
        DEVICE,
        t_min_search,
        t_max_search,
        num_points=num_points_search,
        threshold=threshold_search
    )

    if potential_zeros:
        print("\nValores de 't' para zeros potenciais encontrados (Re(s)=0.5):")
        # Agrupar zeros próximos para evitar múltiplas listagens do mesmo zero "largo"
        if len(potential_zeros) > 1:
            grouped_zeros = []
            current_group_start = potential_zeros[0]
            for i in range(1, len(potential_zeros)):
                # Se a diferença para o anterior for maior que um pequeno delta (resolução da busca), consideramos um novo grupo
                if potential_zeros[i] - potential_zeros[i - 1] > (
                        t_max_search - t_min_search) / num_points_search * 5:  # 5x a resolução
                    grouped_zeros.append((current_group_start + potential_zeros[i - 1]) / 2)  # Média do grupo
                    current_group_start = potential_zeros[i]
            grouped_zeros.append((current_group_start + potential_zeros[-1]) / 2)  # Último grupo

            print("Zeros agrupados (média de t para cada grupo):")
            for t_val in grouped_zeros:
                print(f"  t ≈ {t_val:.6f}")
        else:  # Apenas um zero potencial ou nenhum
            for t_val in potential_zeros:
                print(f"  t ≈ {t_val:.6f}")

    else:
        print("\nNenhum zero potencial encontrado com os parâmetros atuais.")
        print("Sugestões:")
        print("  - Aumente o 'threshold_search' se o modelo não for preciso o suficiente.")
        print("  - Aumente 'num_points_search' para uma busca mais fina.")
        print("  - Verifique o intervalo [t_min_search, t_max_search].")
        print("  - Avalie a precisão do seu modelo (erro de validação).")

    end_time_zeros = time()
    print(f"\nTempo total de execução da busca de zeros: {(end_time_zeros - start_time_zeros):.2f} segundos")


# ### PONTO DE ENTRADA PRINCIPAL DO SCRIPT ###
if __name__ == "__main__":
    print("=" * 70)
    print("ZetaNet - Treinamento e Busca de Zeros da Função Zeta de Riemann")
    print("=" * 70)

    while True:
        print("\nO que você gostaria de fazer?")
        print("  1. Treinar o modelo ZetaNet")
        print("  2. Buscar zeros com um modelo ZetaNet treinado")
        print("  3. Sair")
        action = input("Escolha uma opção (1, 2 ou 3): ").strip()

        if action == '1':
            print("\n--- Iniciando Processo de Treinamento do Modelo ---")
            main_train()
            break
        elif action == '2':
            print("\n--- Iniciando Processo de Busca de Zeros ---")
            main_find_zeros()
            break
        elif action == '3':
            print("Saindo do programa.")
            break
        else:
            print("Opção inválida. Por favor, digite 1, 2 ou 3.")