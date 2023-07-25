#!/bin/bash
base_dir='results/2023-07-24/transfer_multi_eval'
other_dir='results/2023-07-25/transfer_multi_eval'

curr=REP_GDSC
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

curr=REP_CTD2
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

curr=GDSC_REP
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

curr=GDSC_CTD2
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

curr=CTD2_REP
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

curr=CTD2_REP
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/train_avg.pkl $base_dir/$curr 
rsync -avzP --append --progress meisterm@juno-xfer01.mskcc.org:/work/tansey/meisterm/$base_dir/$curr/test_avg.pkl $base_dir/$curr 

