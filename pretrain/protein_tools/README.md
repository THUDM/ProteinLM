
# Megatron-Protein

This directory contains scripts that preprocess TAPE's pfam dataset to the format required by Megatron-LM.

## Step 1: pfam to json

Megaton-LM requires training data in the json format, with one json containing a text sample per line.

Running  [`pfam2json.sh`](./pfam2json.sh) transforms the lmdb file to json file. It contains two steps:

* lmdb to tab separated file
* tab separated file to json file

You will get something like:

```
{"text": "G C T V E D R C L I G M G A I L L N G C V I G S G S L V A A G A L I T Q "}
{"text": "A D G I N L E I P R G E W I S V I G G N G S G K S T F L K S L I R L E A V K K G R I Y L E G R E L K K W S D R T L Y E K A G F V F Q N P E L Q F I R D T V F D E I A F G A R Q R S W P E E Q V E R K T A E L L Q E F G L D G H Q K A H P F T L S L G Q K R R L S V A T M L L F D Q D L L L L D E P T F "}
```

## Step 2: json to binary file

By running [`preprocess_tape.sh`](./preprocess_tape.sh),
the json file is then processed into a binary format for training.

You will see two files `tape_pfam_all_text_document.bin` and `tape_pfam_all_text_document.idx` after preprocessing.
