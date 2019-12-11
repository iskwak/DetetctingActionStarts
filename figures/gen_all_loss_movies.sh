
IN_DIR=/nrs/branson/kwaki/outputs/M174/hoghof/
OUT_DIR=/groups/branson/bransonlab/kwaki/plotting/174_all

EXP_DIR=${IN_DIR}/20190405-perframe_stop_0.5-perframe_0.99-loss_hungarian-learning_rate_1e-05-hantman_tp_4.0-hantman_fp_1.0-hantman_fn_2.0-decay_step_5-decay_0.9-anneal_type_exp_step/predictions/test/
EXPS=${EXP_DIR}*

i=0
for EXP in ${EXPS}
do
    # if [ $i -lt 270 ]; then
    # if [ $(($i % 5)) == 0 -a $i -lt 190 ]; then
    # if [ $(($i % 5)) == 0 ]; then
        # echo ${EXP##*/}
        python -m figures.make_all_loss_movies --input_dir ${IN_DIR} --output_dir ${OUT_DIR} --experiment ${EXP##*/}
    # fi
    i=$((${i}+1))
done


# # M173
# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M173/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M173_20150430_v031

# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M173/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M173_20150506_v007

# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M173/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M173_20150506_v044

# # M174
# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M174/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M174_20150417_v077

# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M174/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M174_20150427_v021

# python -m figures.make_all_loss_movies --input_dir /nrs/branson/kwaki/outputs/M174/hoghof/ --output_dir /groups/branson/bransonlab/kwaki/plotting/ --experiment M174_20150501_v025
