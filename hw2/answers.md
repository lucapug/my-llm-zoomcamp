q1:

ollama -v
0.1.48

q2:
{"schemaVersion":2,"mediaType":"application/vnd.docker.distribution.manifest.v2+json","config":{"mediaType":"application/vnd.docker.container.image.v1+json","digest":"sha256:887433b89a901c156f7e6944442f3c9e57f3c55d6ed52042cbb7303aea994290","size":483},"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:c1864a5eb19305c40519da12cc543519e48a0697ecd30e15d5ac228644957d12","size":1678447520},{"mediaType":"application/vnd.ollama.image.license","digest":"sha256:097a36493f718248845233af1d3fefe7a303f864fae13bc31a3a9704229378ca","size":8433},{"mediaType":"application/vnd.ollama.image.template","digest":"sha256:109037bec39c0becc8221222ae23557559bc594290945a2c4221ab4f303b8871","size":136},{"mediaType":"application/vnd.ollama.image.params","digest":"sha256:22a838ceb7fb22755a3b0ae9b4eadde629d19be1f651f73efb8c6b4e2cd0eea0","size":84}]}

q3:

docker exec -it ollama ollama run gemma:2b

Sure, here is the answer:

10 * 10 = 100

q4:

/home/lucapug/my-llm-zoomcamp/hw2/ollama_files

cd /home/lucapug/my-llm-zoomcamp/hw2/

docker run -it
--rm
-v ./ollama_files:/root/.ollama
-p 11434:11434
--name ollama
ollama/ollama

du -h

1.6G    ./models/blobs
8.0K    ./models/manifests/registry.ollama.ai/library/gemma
12K     ./models/manifests/registry.ollama.ai/library
16K     ./models/manifests/registry.ollama.ai
20K     ./models/manifests
1.6G    ./models
1.6G    .

q5:
Dockerfile

####begin

FROM ollama/ollama

COPY ./ollama_files /root/.ollama

####end

q6: 
docker build -t ollama-gemma2b .

docker run -it --rm -p 11434:11434 ollama-gemma2b

execution of q6.ipynb

total_tokens=315
