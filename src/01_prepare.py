import pandas as pd
from pathlib import Path
from zipfile import ZipFile

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

ZIP_NAME = "microdados_enem_2023.zip"
CSV_NAME_INSIDE = "DADOS/MICRODADOS_ENEM_2023.csv"
ZIP_PATH = DATA_RAW / ZIP_NAME

SAMPLE_SIZE = 50_000 
CHUNKS_TO_READ = 2

TARGETS = ["NU_NOTA_MT", "NU_NOTA_CN", "NU_NOTA_LC", "NU_NOTA_CH", "NU_NOTA_REDACAO"]
FEATURES = [
    "Q006", "Q002", "TP_ESCOLA", "TP_COR_RACA",
]
USECOLS = sorted(list(set(TARGETS + FEATURES)))

def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Arquivo ZIP não encontrado em: {ZIP_PATH}. Verifique o caminho e o nome do arquivo.")
    
    print(f"Lendo dados do ZIP: {ZIP_PATH}")
    
    chunks = []
    
    print(f"Lendo os primeiros {CHUNKS_TO_READ} pedaços (chunks) do CSV para criar a amostra...")
    with ZipFile(ZIP_PATH, "r") as z:
        with z.open(CSV_NAME_INSIDE) as f:
            chunk_iterator = pd.read_csv(
                f, sep=";", encoding="iso-8859-1",
                chunksize=200_000, low_memory=False
            )
 
            first_chunk = next(chunk_iterator)
            print("--- COLUNAS ENCONTRADAS NO ARQUIVO CSV ORIGINAL ---")
            print(first_chunk.columns.tolist())
            print("-------------------------------------------------")

            chunks.append(first_chunk)
            
            for i, chunk in enumerate(chunk_iterator):
                if i >= CHUNKS_TO_READ - 1: 
                    break
                chunks.append(chunk)

    if not chunks:
        raise RuntimeError("Nenhum dado foi carregado.")

    df_from_chunks = pd.concat(chunks, ignore_index=True)
    
    cols_existentes = [col for col in USECOLS if col in df_from_chunks.columns]
    print(f"\nColunas relevantes que foram encontradas: {cols_existentes}")
    df_filtered = df_from_chunks[cols_existentes]

    if "NU_NOTA_REDACAO" in df_filtered.columns:
        df_filtered = df_filtered.rename(columns={"NU_NOTA_REDACAO": "NU_NOTA_RED"})
   #primeira etapa     
    if len(df_filtered) < SAMPLE_SIZE:
        df_sample = df_filtered
    else:
        df_sample = df_filtered.sample(n=SAMPLE_SIZE, random_state=42)
    
    print(f"Amostra de {len(df_sample)} linhas criada com sucesso.")
    
    nota_cols_presentes = [c for c in TARGETS if c in df_sample.columns]
    for col in nota_cols_presentes:
        df_sample[col] = pd.to_numeric(df_sample[col], errors='coerce')
        
    df_sample.dropna(subset=nota_cols_presentes, how='all', inplace=True)
    
    parquet_path = DATA_PROCESSED / f"enem_2023_amostra_{SAMPLE_SIZE}.parquet"
    df_sample.to_parquet(parquet_path, index=False)
    
    print(f"Arquivo de amostra processado e salvo com {len(df_sample)} linhas em: {parquet_path}")

if __name__ == "__main__":
    main()