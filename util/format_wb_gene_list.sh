#!/bin/bash

###################################################################
#Script Name	:                                                                                           
#Description	: 
#Args           	:                                                                                           
#Author       	:Christopher Brittin
#Date           :2019-12-05                                                
#Email         	:"cabritin" <at> "gmail." "com"                                          
###################################################################

dbnum=$1
fout=$2
tmpdir=__gene_tmp__
header="id,gene_short_name"

base_file=https://downloads.wormbase.org/releases/$dbnum/ONTOLOGY/gene_association.$dbnum.wb.c_elegans

echo Downloading $base_file

curl -vs $base_file 2>&1 | grep WB | awk -F'\t' '{print $2","$3}' | sort -u >> $fout

