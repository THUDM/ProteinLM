set -xe

PFAM=<Specify path form PFAM>
VOCAB=<Specify path form VOCAB>

python ../tools/preprocess_data.py --input $PFAM/pfam_all.json \
	--tokenizer-type BertWordPieceCase --vocab-file $VOCAB/iupac_vocab.txt \
	--output-prefix $PFAM/tape_pfam_all --dataset-impl mmap --workers 64
