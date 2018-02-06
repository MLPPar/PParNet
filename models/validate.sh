#!/bin/sh
# path to moses decoder: https://github.com/moses-smt/mosesdecoder
mosesdecoder=/mnt/gna0/rsennrich/tools/mosesdecoder
# mosesdecoder=/path/to/mosesdecoder
# subword_nmt=/path/to/subword-nmt 

# path to nematus ( https://www.github.com/rsennrich/nematus )
nematus=/fs/zisa0/mbehnke/dev-nematus

dir=/fs/zisa0/mbehnke/mlp_project

test=newstest2016.bpe.tr
ref=$dir/data/newstest2016.tok.en

scripts=$(cd `dirname $0` && pwd)

$scripts/postprocess-test.sh < $test.translated > $test.translated.postprocessed

## get BLEU
BEST=`cat $scripts/bleu || echo 0`
$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $test.translated.postprocessed >> $scripts/../bleu
BLEU=`$mosesdecoder/scripts/generic/multi-bleu.perl $ref < $test.translated.postprocessed | cut -f 3 -d ' ' | cut -f 1 -d ','`
BETTER=`echo "$BLEU > $BEST" | bc`

echo "BLEU = $BLEU"
echo $BLEU > $scripts/bleu

