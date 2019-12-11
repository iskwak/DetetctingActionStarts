"""Script to run bsub jobs."""
import os
import sys
import time
import stat
import re
import argparse


def main(base_command, main_params, output_dir, host_flag):
    # first setup the output space.
    # hard code the search....
    # learning_rates = [0.001, 0.005, 0.0001]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    queues = [
        'gpu_v100',
        'gpu_p100',
        'gpu_1080',
        'local'
    ]
    # loop over leanring rates
    for queue in queues:
            # construct the output directory name.
            output_name =\
                time.strftime('%Y%m%d') + '-' + queue
            full_path = os.path.join(output_dir, output_name)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

            # setup the new parameters, specific to this sweep.
            main_params['out_dir'] = full_path
            new_command = parse_line(base_command, main_params)

            # construct the bsub script, it will be generated and placed inside
            # of the output folder.
            bsub_script = construct_bsub_script(full_path)

            # now construct the singularity script, which is called by the bsub
            # script.
            construct_singularity_script(full_path, new_command)

            # construct the bsub command and execute.
            bsub_command = create_bsub_command(full_path, bsub_script, queue)
            # print(bsub_command)
            # save the command to the output directory.
            main_script = write_command_script(full_path, bsub_command)
            # execute the command.
            if queue != 'local':
                os.system(main_script)
            # else, run it manually on the desktop
            # if host_flag == "local":
            #     # run the busb script
            #     # import pdb; pdb.set_trace()
            #     os.system(bsub_script)
            # else:
            #     os.system(main_script)


def parse_line(line, keyval_dict):
    """Parse the line."""
    results = re.findall('\%([a-zA-Z0-9_]+)\%', line)
    if results:
        # make sure the key exists
        for i in range(len(results)):
            # print(results[i])
            if results[i] not in keyval_dict.keys():
                raise ValueError(
                    'Key ' + results[i] + ' not in provided dictionary')
            line = re.sub(
                '%' + results[i] + '%',
                str(keyval_dict[results[i]]),
                line
            )
    return line


def write_command_script(full_path, bsub_command):
    """Write the actual bsub command to file."""
    out_name = os.path.join(full_path, 'command.sh')
    with open(out_name, 'w') as out:
        out.write(bsub_command)
    # make the new script exectuable.
    st = os.stat(out_name)
    os.chmod(out_name, st.st_mode | stat.S_IEXEC)
    return out_name


def construct_script(input_filename, output_filename, keyval_dict):
    # assume that bsub_script.sh is the current working directory
    with open(input_filename, 'r') as base_script:
        with open(output_filename, 'w') as new_script:
            for line in base_script:
                parsed = parse_line(line, keyval_dict)
                new_script.write(parsed)
    # make the new script exectuable.
    st = os.stat(output_filename)
    os.chmod(output_filename, st.st_mode | stat.S_IEXEC)


def construct_singularity_script(full_path, python_command):
    """Construct the main bsub script for this run."""
    # for now hard code this... hard code the bsub stuff.
    # process_script_name = os.path.join(full_path, 'singularity_script.sh')
    keyval_dict = {
        'path': full_path,
        'python_command': python_command
    }
    input_script_name = 'singularity_script.sh'
    new_script_name = os.path.join(full_path, 'singularity_script.sh')
    construct_script(input_script_name, new_script_name, keyval_dict)


def construct_bsub_script(full_path):
    """Construct the main bsub script for this run."""
    # for now hard code this... hard code the bsub stuff.
    process_script_name = os.path.join(full_path, 'singularity_script.sh')
    keyval_dict = {
        'image': '/misc/local/singularity/branson.simg',
        'command': process_script_name
    }

    # assume the template script is in the base folder.
    input_script_name = 'bsub_script.sh'
    new_script_name = os.path.join(full_path, 'bsub_script.sh')
    construct_script(input_script_name, new_script_name, keyval_dict)

    return new_script_name


def create_bsub_command(full_path, bsub_script, queue):
    """Construct the bsub command."""
    bsub_command = (
        'bsub '
        '-n 4 '          # num cpus
        '-gpu "num=1" '  # num gpus
        '-q %queue% '  # gpu queue
        '-o %out_log% '  # output log location
        '%command%'      # command to be run
    )

    bsub_dict = {
        'out_log': os.path.join(full_path, 'output.log'),
        'command': bsub_script,
        'queue': queue
    }

    bsub_command = parse_line(bsub_command, bsub_dict)

    return bsub_command


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', action='store')
    args = parser.parse_args()

    return args.host


def create_python_command():
    """Construct the base command."""
    # initially hard coded... can only sweep over learning rate and
    # hidden dimensions... How to generalize to feed forward networks?
    base_command, main_params, output_dir =\
        hungarianmouse_setup()
    # base_command, main_params, learning_rates, hidden_dims, output_dir =\
    #     mpii_setup()

    return base_command, main_params, output_dir


def hungarianmouse_setup():
    """Setup the hungarian mouse parameters."""
    base_command = (
        'python timed_hungarianmouse.py '
        '--train_file %train_file% '
        '--test_file %test_file% '
        '--val_file %val_file% '
        '--arch %arch% '
        '--feat_keys %feat_keys% '
        '--out_dir %out_dir% '
        '--learning_rate %learning_rate% '
        '--lstm_hidden_dim %lstm_hidden_dim% '
        '--image_dir /localhome/kwaki/frames '
        '--cuda_device 0 '
        '--hantman_mini_batch=10 '
        '--hantman_perframeloss=WEIGHTED_MSE '
        '--hantman_seq_length=1500 '
        '--total_epochs=100 '
        '--hantman_perframe_weight=100.0 '
        '--hantman_struct_weight=1.0 '
        '--hantman_tp=10.0 '
        '--hantman_fp=0.25 '
        '--hantman_fn=20.0 '
        '--reweight --normalize'
    )

    # # main parameters
    # main_params = {
    #     'train_file': '/nrs/branson/kwaki/data/20180212_3dconv/m173_train.hdf5',
    #     'test_file': '/nrs/branson/kwaki/data/20180212_3dconv/m173_test.hdf5',
    #     'val_file': '/nrs/branson/kwaki/data/20180212_3dconv/m173_val.hdf5',
    #     'arch': 'bidirconcat',
    #     'feat_keys': 'conv3d,pos',
    #     'out_dir': '',
    #     'learning_rate': '',
    #     'lstm_hidden_dim': ''
    # }
    # # main parameters
    main_params = {
        'train_file': '/nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_train.hdf5',
        'val_file': '/nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_valid.hdf5',
        'test_file': '/nrs/branson/kwaki/data/20180410_all_hoghof/all_mouse_multi_day2_test.hdf5',
        'arch': 'bidirconcat',
        'feat_keys': 'hoghof,pos_features',
        'out_dir': '',
        'learning_rate': '0.0001',
        'lstm_hidden_dim': '256'
    }

    output_dir = '/nrs/branson/kwaki/outputs/20180424_hoghof_all_timing'

    return base_command, main_params, output_dir


if __name__ == "__main__":
    # setup the command variables
    host_flag = parse_args(sys.argv)

    base_command, main_params, output_dir =\
        create_python_command()
    main(
        base_command, main_params, output_dir, host_flag)
