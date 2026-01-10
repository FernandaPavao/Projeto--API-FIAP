"""
Script Principal - Tech Challenge Fase 4
Executa todo o pipeline de coleta e pré-processamento
Pessoa 1: Coleta e Pré-processamento dos Dados
"""

import sys
import os
import subprocess
import time
from datetime import datetime

def print_header(text):
    """Imprime cabeçalho formatado"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def print_step(step_num, total_steps, description):
    """Imprime informação do passo atual"""
    print(f"\n{'─'*70}")
    print(f"📍 PASSO {step_num}/{total_steps}: {description}")
    print(f"{'─'*70}\n")

def run_script(script_path, description):
    """
    Executa um script Python e verifica o status
    
    Args:
        script_path (str): Caminho do script
        description (str): Descrição do que o script faz
    
    Returns:
        bool: True se executou com sucesso, False caso contrário
    """
    print(f"⏳ Executando: {description}...")
    print(f"   Script: {script_path}\n")
    
    start_time = time.time()
    
    try:
        # Executar o script
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False,
            text=True
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n✅ Concluído em {elapsed_time:.2f} segundos")
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ Erro ao executar {script_path}")
        print(f"   Tempo decorrido: {elapsed_time:.2f} segundos")
        print(f"   Código de saída: {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Erro inesperado: {str(e)}")
        return False

def check_files_exist():
    """Verifica se todos os arquivos necessários foram gerados"""
    print_header("VERIFICAÇÃO DE ARQUIVOS GERADOS")
    
    required_files = [
        'data/raw/VALE3_SA_raw.csv',
        'data/processed/VALE3_SA_processed.csv',
        'data/processed/train_data.npz',
        'data/processed/val_data.npz',
        'data/processed/test_data.npz',
        'data/processed/scaler.pkl',
        'data/processed/data_info.json'
    ]
    
    all_exist = True
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"   ✅ {file_path} ({size:.2f} KB)")
        else:
            print(f"   ❌ {file_path} - NÃO ENCONTRADO")
            all_exist = False
    
    return all_exist

def display_summary():
    """Exibe resumo final do pipeline"""
    print_header("RESUMO DO PIPELINE")
    
    print("📦 ARQUIVOS GERADOS PARA A PESSOA 2:\n")
    print("   1️⃣  data/processed/train_data.npz")
    print("       → Dados de treino (X_train, y_train)")
    print("")
    print("   2️⃣  data/processed/val_data.npz")
    print("       → Dados de validação (X_val, y_val)")
    print("")
    print("   3️⃣  data/processed/test_data.npz")
    print("       → Dados de teste (X_test, y_test)")
    print("")
    print("   4️⃣  data/processed/scaler.pkl")
    print("       → Scaler para desnormalizar previsões")
    print("")
    print("   5️⃣  data/processed/data_info.json")
    print("       → Metadados do dataset (features, shapes, etc.)")
    print("")
    
    print("📊 ANÁLISES E VISUALIZAÇÕES:\n")
    print("   • reports/figures/*.png - Gráficos de análise")
    print("   • reports/figures/summary_statistics.txt - Estatísticas")
    print("")
    
    print("📝 DOCUMENTAÇÃO:\n")
    print("   • README.md - Documentação completa do projeto")
    print("")

def main():
    """
    Função principal que executa todo o pipeline
    """
    # Banner inicial
    print("\n" + "="*70)
    print("║" + " "*68 + "║")
    print("║" + " "*15 + "TECH CHALLENGE FASE 4" + " "*32 + "║")
    print("║" + " "*10 + "Pipeline de Coleta e Pré-processamento" + " "*21 + "║")
    print("║" + " "*68 + "║")
    print("="*70)
    print(f"\n🕐 Início: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"💼 Empresa: Vale S.A. (VALE3.SA)")
    print(f"📅 Período: Últimos 5 anos")
    
    total_steps = 4
    pipeline_start = time.time()
    
    # Lista de scripts a serem executados
    scripts = [
        {
            'path': 'src/data_collection.py',
            'description': 'Coleta de Dados (yfinance)',
            'required': True
        },
        {
            'path': 'src/data_preprocessing.py',
            'description': 'Pré-processamento e Indicadores Técnicos',
            'required': True
        },
        {
            'path': 'src/data_preparation.py',
            'description': 'Preparação para LSTM (Normalização e Janelas)',
            'required': True
        },
        {
            'path': 'src/eda_analysis.py',
            'description': 'Análise Exploratória de Dados (EDA)',
            'required': False  # Opcional, não bloqueia o pipeline
        }
    ]
    
    # Executar cada script
    success = True
    for i, script in enumerate(scripts, 1):
        print_step(i, total_steps, script['description'])
        
        result = run_script(script['path'], script['description'])
        
        if not result and script['required']:
            print(f"\n❌ Pipeline interrompido devido a erro no passo {i}")
            success = False
            break
        elif not result:
            print(f"\n⚠️  Aviso: Passo {i} falhou, mas não é obrigatório. Continuando...")
    
    # Tempo total
    total_time = time.time() - pipeline_start
    
    if success:
        # Verificar arquivos
        files_ok = check_files_exist()
        
        # Resumo final
        display_summary()
        
        # Mensagem de sucesso
        print_header("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
        print(f"⏱️  Tempo total: {total_time:.2f} segundos ({total_time/60:.2f} minutos)")
        print(f"🎉 Todos os dados estão prontos para a Pessoa 2!")
        print("")
        print("📋 PRÓXIMOS PASSOS:")
        print("   1. Revisar os gráficos em 'reports/figures/'")
        print("   2. Verificar 'data/processed/data_info.json'")
        print("   3. Passar os arquivos para a Pessoa 2 treinar o modelo LSTM")
        print("")
        print("="*70 + "\n")
        
        return 0
    else:
        print_header("❌ PIPELINE FALHOU")
        print(f"⏱️  Tempo até a falha: {total_time:.2f} segundos")
        print("")
        print("🔍 SOLUÇÃO:")
        print("   1. Verifique a mensagem de erro acima")
        print("   2. Corrija o problema")
        print("   3. Execute novamente: python run_pipeline.py")
        print("")
        print("="*70 + "\n")
        
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)