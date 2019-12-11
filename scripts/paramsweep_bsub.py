"""Script to run bsub jobs."""
import os
import sys
import time
import stat
import re
import argparse
import paramsweep_bsub_helper


def sweep_options(base_command, output_dir, main_params, tunable_params, host_flag):
    """Recursive function to sweep options and create the run commands."""
    all_keys = tunable_params.keys()
    all_keys.sort()

    # start the command, each recursive function will update the param_dict
    # which will be passed into the construct script commands.
    param_dict = main_params
    # the sweep options code also needs to build the output directory.
    output_dir = os.path.join(output_dir, time.strftime('%Y%m%d'))
    sweep_options_helper(
        base_command, output_dir, param_dict, all_keys, tunable_params,
        host_flag)


def sweep_options_helper(base_command, output_dir, param_dict, all_keys, tunable_params, host_flag):
    """Helper recursive function for building and running commands."""
    # first check to see if the list of keys is empty.
    if not all_keys:
        # if the keys are empty, then time to build the command and maybe run
        # it.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # update the param_dict to have the output directory.
        param_dict["out_dir"] = output_dir
        construct_run_script(base_command, output_dir, param_dict, all_keys, host_flag)
        return

    base_out = output_dir
    key = all_keys[-1]
    sub_keys = all_keys[:-1]
    for value in tunable_params[key]:
        param_dict[key] = value
        output_dir = base_out + "-" + key + "_" + str(value)

        # call the the recursive helper with one less key (was popped)
        sweep_options_helper(
            base_command, output_dir, param_dict, sub_keys, tunable_params, host_flag)
        print("key val: %s, %s" %  (key, str(value)))
        # import pdb; pdb.set_trace()


def construct_run_script(base_command, full_path, param_dict, all_keys, host_flag):
    """Constructs/write to disk and run scripts."""
    # create the command using the param dict
    new_command = parse_line(base_command, param_dict)

    # construct the bsub script, it will be generated and placed inside
    # of the output folder.
    bsub_script = construct_bsub_script(full_path)

    # now construct the singularity script, which is called by the bsub
    # script.
    construct_singularity_script(full_path, new_command)

    # construct the bsub command and execute.
    bsub_command = create_bsub_command(full_path, bsub_script)
    # print(bsub_command)
    # save the command to the output directory.
    main_script = write_command_script(full_path, bsub_command)
    # execute the command.
    if host_flag == "local":
        # run the busb script
        # os.system(bsub_script)
        print(bsub_script)
    else:
        print(main_script)
        os.system(main_script)


def main(base_command, main_params, tunable_params, output_dir, host_flag):
    # first setup the output space.
    # hard code the search....
    # learning_rates = [0.001, 0.005, 0.0001]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # loop over the tunable params and create all combos.
    sweep_options(base_command, output_dir, main_params, tunable_params, host_flag)


def create_output_folder(tunable_params):
    """Creates an output folder name, based on the tunable parameters."""
    # base name starts with the current time.
    output_name = time.strftime('%Y%m%d') + str(tunable_params[0])

    for i in range(1, len(tunable_params)):
        output_name = output_name + '-' + str(tunable_params[i])

    return output_name


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
        # 'image': '/misc/local/singularity/branson_v3.simg',
        'image': '/misc/local/singularity/branson_cuda10_27.simg',
        # 'image': '/nrs/branson/kwaki/simgs/base_python35.simg',
        'command': process_script_name
    }

    # assume the template script is in the base folder.
    input_script_name = 'bsub_script.sh'
    new_script_name = os.path.join(full_path, 'bsub_script.sh')
    construct_script(input_script_name, new_script_name, keyval_dict)

    return new_script_name


def create_bsub_command(full_path, bsub_script):
    """Construct the bsub command."""
    # bsub_command = (
    #     'bsub '
    #     '-n 10 '                                     # num cpus
    #     # '-n 4 '                                     # num cpus
    #     '-gpu "num=1" '  # num gpus
    #     # '-gpu "num=1:gmodel=TeslaV100_SXM2_32GB" '  # num gpus
    #     '-q gpu_tesla '                             # gpu queue
    #     # '-q gpu_rtx '                             # gpu queue
    #     # '-m f13u27 '
    #     '-gpu "num=1" '                             # num gpus
    #     '-o %out_log% '                             # output log location
    #     '%command%'                                 # command to be run
    # )
    bsub_command = (
        'bsub '
        '-n 5 '                                     # num cpus
        # '-gpu "num=1:gmodel=TeslaV100_SXM2_32GB" '  # num gpus
        '-q gpu_rtx '                             # gpu queue
        # '-m e11u15 '
        # '-m f15u10 '
        # '-q gpu_any '                             # gpu queue
        # '-q gpu_gtx '                             # gpu queue
        # '-m f13u15 '
        # '-q gpu_any '                             # gpu queue
        # '-m f14u05 '
        '-gpu "num=1" '                             # num gpus
        '-o %out_log% '                             # output log location
        '%command%'                                 # command to be run
    )

    bsub_dict = {
        'out_log': os.path.join(full_path, 'output.log'),
        'command': bsub_script
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
    base_command, main_params, tunable_params, output_dir =\
        paramsweep_bsub_helper.hungarianmouse_setup()
    # base_command, main_params, tunable_params, output_dir =\
    #     paramsweep_bsub_helper.hantman_2dconv()
    # base_command, main_params, tunable_params, output_dir =\
    #     paramsweep_bsub_helper.feature_creation()
    # base_command, main_params, tunable_params, output_dir =\
    #     paramsweep_bsub_helper.hantman_3dconv()

    return base_command, main_params, tunable_params, output_dir


if __name__ == "__main__":
    # setup the command variables
    host_flag = parse_args(sys.argv)

    base_command, main_params, tunable_params, output_dir =\
        create_python_command()
    main(base_command, main_params, tunable_params, output_dir, host_flag)
