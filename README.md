---

# Trace Clustering

![Esquema do Projeto](./esquema_grafico-trace_clustering.png?raw=true)

## Sobre o Projeto

O **Trace Clustering** é um trabalho acadêmico focado em mineração de processos, utilizando algoritmos de agrupamento (clustering) para analisar e extrair padrões de logs de eventos. O projeto oferece scripts para pré-processamento dos dados, geração de matrizes de features e aplicação do K-Means, além de uma interface web (Django) para visualização dos resultados.

---

## Estrutura do Projeto

- **datasets/**  
  Contém logs de eventos utilizados como entrada (ex: `incident_log.csv`).

- **src/**  
  Scripts para processamento e análise dos dados:
  - `calc_pm_metrics.py`: Calcula métricas de mineração de processos.
  - `create_matrices.py`: Gera matrizes de features a partir dos dados brutos.
  - `kmeans.py`: Executa o algoritmo de clustering K-Means.
  - `utils/`: Funções auxiliares.

- **matrices/**  
  Matrizes de dados utilizadas nos algoritmos de clustering (`individual_specialist_binary.csv`, `tf.csv`, `tfidf.csv`).

- **kmeans_results/**  
  Resultados dos agrupamentos, agrupamentos intermediários e estatísticas de execução (`metadata_*.csv`, subdiretórios para diferentes outputs).

- **ic_pm/** & **report_kmeans/**  
  Estrutura do projeto Django para visualização web dos resultados, administração e relatórios.

- **static/**  
  Arquivos estáticos do frontend (CSS, JS, imagens).

- **run.py**  
  Script de inicialização e execução principal.

- **db.sqlite3**  
  Banco de dados do backend Django.

---

## Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/BrunoPontesLira/trace_clustering.git
   cd trace_clustering
   ```

2. **Crie o ambiente virtual e instale as dependências:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # (Linux/Mac)
   venv\Scripts\activate     # (Windows)
   pip install -r requirements.txt
   ```

3. **Configure o banco de dados Django:**
   ```bash
   python manage.py migrate
   ```

---

## Como Usar

- Para processar um log de eventos e aplicar clustering:
  ```bash
  python src/create_matrices.py datasets/incident_log.csv
  python src/kmeans.py matrices/individual_specialist_tfidf.csv
  ```
  Os resultados serão salvos em `kmeans_results/`.

- Para visualizar os resultados e relatórios via interface web:
  ```bash
  python manage.py runserver
  ```
  Acesse [http://localhost:8000](http://localhost:8000) no navegador.

---

## Exemplos de Dados

Os arquivos de exemplo estão em `datasets/`. Você pode usar `incident_log.csv` para testar o pipeline completo.

---

## Contribuição

Contribuições são bem-vindas!  
Abra uma issue para sugestões ou reporte problemas.  
Para contribuir com código, envie um pull request.

---

## Autor

- **Bruno Pontes Lira**  
  [GitHub](https://github.com/BrunoPontesLira)

---

## Licença

Este projeto é distribuído para fins acadêmicos. Entre em contato para outros usos.