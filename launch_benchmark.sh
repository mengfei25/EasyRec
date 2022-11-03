#!/bin/bash
set -xe

function main {
    # set common info
    source common.sh
    init_params $@
    fetch_device_info
    set_environment

    # requirements
    if [ "${CONFIG_DIR}" == "" ] || [ "${DATASET_DIR}" == "" ] || [ "${CKPT_DIR}" == "" ];then
        set +x
        echo "[ERROR] Please set EXAMPLE_ARGS before launch"
        echo "  export CONFIG_DIR=/example/config/use"
        echo "  export DATASET_DIR=/example/data/use"
        echo "  export CKPT_DIR=/example/ckpt/use"
        exit 1
        set -x
    fi
    source ./scripts/gen_proto.sh
    pip install -e .

    # if multiple use 'xxx,xxx,xxx'
    model_name_list=($(echo "${model_name}" |sed 's/,/ /g'))
    batch_size_list=($(echo "${batch_size}" |sed 's/,/ /g'))

    # generate benchmark
    for model_name in ${model_name_list[@]}
    do
        # cache
        python -m easy_rec.python.eval \
            --pipeline_config_path $CONFIG_DIR \
            --dataset_dir $DATASET_DIR \
            --ckpt_dir $CKPT_DIR \
            --precision $precision --batch_size 1 --num_iter 10 \
            ${addtion_options}
        #
        for batch_size in ${batch_size_list[@]}
        do
            # clean workspace
            logs_path_clean
            generate_core
            # launch
            echo -e "\n\n\n\n Running..."
            cat ${excute_cmd_file} |column -t > ${excute_cmd_file}.tmp
            mv ${excute_cmd_file}.tmp ${excute_cmd_file}
            source ${excute_cmd_file}
            echo -e "Finished.\n\n\n\n"
            # get throughput
            for get_fps in $(find ${log_dir} -type f -name "rcpi*.log")
            do
                grep 'Inference Time :' $get_fps |tail -1 |sed 's/.*://;s/[^0-9.]//g' |\
                        awk -v b=$batch_size -v i=$num_iter '{
                            printf("inference Throughput: %.3f samples/s\n", b*i/$1);
                        }' >> ${get_fps}
            done
            # collect launch result
            collect_perf_logs
        done
    done
}

# run
function generate_core {
    # generate multiple instance script
    for(( i=0; i<instance; i++ ))
    do
        real_cores_per_instance=$(echo ${device_array[i]} |awk -F, '{print NF}')
        log_file="${log_dir}/rcpi${real_cores_per_instance}-ins${i}.log"

        # instances
        if [ "${device}" != "cuda" ];then
            OOB_EXEC_HEADER=" numactl -m $(echo ${device_array[i]} |awk -F ';' '{print $2}') "
            OOB_EXEC_HEADER+=" -C $(echo ${device_array[i]} |awk -F ';' '{print $1}') "
        else
            OOB_EXEC_HEADER=" CUDA_VISIBLE_DEVICES=${device_array[i]} "
        fi
        printf " ${OOB_EXEC_HEADER} \
            python -m easy_rec.python.eval \
                --pipeline_config_path $CONFIG_DIR \
                --dataset_dir $DATASET_DIR \
                --ckpt_dir $CKPT_DIR \
                --precision $precision --batch_size $batch_size --num_iter $num_iter \
                ${addtion_options} \
        > ${log_file} 2>&1 &  \n" |tee -a ${excute_cmd_file}
        if [ "${numa_nodes_use}" == "0" ];then
            break
        fi
    done
    echo -e "\n wait" >> ${excute_cmd_file}
}

# download common files
wget -q -O common.sh https://raw.githubusercontent.com/mengfei25/oob-common/main/common.sh

# Start
main "$@"
