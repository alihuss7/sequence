FROM continuumio/miniconda3:24.3.0-0

WORKDIR /app

RUN conda update -n base -c defaults conda -y && \
    conda create -n app-env python=3.11 -y && \
    conda clean -afy

SHELL ["conda", "run", "-n", "app-env", "/bin/bash", "-c"]

RUN conda install -c conda-forge -c bioconda -y \
    anarci \
    hmmer \
    biopython \
    freesasa \
    wget && \
    conda clean -afy

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN bash postBuild

ENV PORT=10000 \
    PYTHONUNBUFFERED=1

EXPOSE 10000

CMD ["conda", "run", "-n", "app-env", "bash", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-10000}"]