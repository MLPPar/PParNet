#/bin/sh

# path to moses decoder: https://github.com/moses-smt/mosesdecoder
# mosesdecoder=/mnt/gna0/rsennrich/tools/mosesdecoder

# suffix of target language files
lng=en

sed 's/\@\@ //g' | \
./detruecase.perl
# $mosesdecoder/scripts/recaser/detruecase.perl
# $mosesdecoder/scripts/tokenizer/detokenizer.perl -l $lng
