set -xe

PFAM=<Specify path form PFAM>

for data in valid holdout train
do
	python lmdb2tab.py --lmdbfile $PFAM/pfam_$data.lmdb --tabfile $PFAM/pfam_$data.tab
	awk '{gsub(/./,"& ",$2); print "{\"text\": \""$2"\"}"}' $PFAM/pfam_$data.tab > $PFAM/pfam_$data.json
done

cat $PFAM/pfam_train.json $PFAM/pfam_valid.json $PFAM/pfam_holdout.json  > $PFAM/pfam_all.json