# script to extract last xcsf pkl file from given (gs, sp) exp dirs
gs=12
sp=0.5

declare -A end_ga_calls=( [4]=14000 [8]=42000 [12]=84000 )

if [ ${sp} = 0 ]; then
    basedir="./frozen/detrm/gs_${gs}/"
else
    basedir="./frozen/stoca/gs_${gs}_sp_${sp}/"
fi

curr_dir=$(pwd)

for dir in ${basedir}/*/; do
    cd $dir
    pwd
    exp_num=$(basename $dir)
    ga_calls="${end_ga_calls[$gs]}"
    tar -xf xcsfs.tar.xz "${exp_num}/xcsf_ga_calls_${ga_calls}.pkl" --strip-components=1
    cd $curr_dir
done
