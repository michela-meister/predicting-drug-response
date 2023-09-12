#!/bin/bash
codeDir="/work/tansey/meisterm/code"
kMax="80"
sMax="20"
mMax="10"

baseDir="/work/tansey/meisterm/results/2023-08-07/opt_k_target_only"

python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_REP" kMax="$kMax" sMax="$sMax" mMax="$mMax"
python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_GDSC" kMax="$kMax" sMax="$sMax" mMax="$mMax"
python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_CTD2" kMax="$kMax" sMax="$sMax" mMax="$mMax"

#baseDir="/work/tansey/meisterm/results/2023-08-07/opt_k_transfer"

#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_REP_GDSC" kMax="$kMax" sMax="$sMax" mMax="$mMax"
#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_REP_CTD2" kMax="$kMax" sMax="$sMax" mMax="$mMax"

#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_GDSC_REP" kMax="$kMax" sMax="$sMax" mMax="$mMax"
#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_GDSC_CTD2" kMax="$kMax" sMax="$sMax" mMax="$mMax"

#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_CTD2_REP" kMax="$kMax" sMax="$sMax" mMax="$mMax"
#python3 "$codeDir/"digest_opt_k.py resultsDir="$baseDir/log_CTD2_GDSC" kMax="$kMax" sMax="$sMax" mMax="$mMax"

