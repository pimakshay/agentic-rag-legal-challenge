import re

with open('tmp_akshay_pipeline.py', 'r') as f:
    akshay_code = f.read()

with open('retrieval/legal_hybrid_rag_pipeline.py', 'r') as f:
    main_code = f.read()

# We need to construct the target pipeline code by hand-picking sections or we can just template it.
