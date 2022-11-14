import os


def get_files_in_dir(source_dir: str, source_format=None):
    list = []
    for path, sub_dirs, files in os.walk(source_dir):
        for name in files:
            if source_format is not None and name.endswith(source_format) and name[0] != '.':
                list.append(os.path.join(path, name))
            elif name[0] != '.':
                list.append(os.path.join(path, name))
    return list

    
def split_filename(full_path: str) -> dict:
    """
    Splits a file name's full path into it parts
    :param full_path:
    :return: a dict with each of the parts includes as key-value pairs
    """
    dir_name, filename = os.path.split(full_path)
    basename, extension = filename.split('.', 1)
    return {'dir_name': dir_name, 'filename': filename, 'basename': basename, 'extension': extension}


def ffmpeg(full_path_infile: str, full_path_outfile: str, log_file: str = None):
    """
    converts a mp4 file into a wave file and records the execution in the log file
    :param full_path_infile:
    :param full_path_outfile:
    """
    try:
        if log_file is None:
            cmd = 'ffmpeg -n -i  {0} -ar 16000 {1} >/dev/null 2>&1'.format(full_path_infile, full_path_outfile)
        else:
            cmd = 'ffmpeg -n -i  {0} -ar 16000 {1} 2> {2}'.format(full_path_infile, full_path_outfile, log_file)
    except Exception as e:
        print("ffmpeg error: " + e)
        
    os.system(cmd)


def ffmpeg_timestamps(full_path_infile: str, full_path_outfile: str, start, end, log_file_name):
    """
    converts a mp4 to a wav with starting and ending timestamps
    :param full_path_infile:
    :param full_path_outfile:
    :param start
    :param end
    """
    cmd = "ffmpeg -n -i {infile} -ss {start} -to {end} -ar 16000 {outfile} 2> {log_file}".format(infile=full_path_infile, start=start, end=end, log_file=log_file_name, outfile=full_path_outfile)
    os.system(cmd)
    
    
def mkdir_if_dne(target_dir):
    # checking if the directory demo_folder exist or not.
    if not os.path.exists(target_dir):
	    # if the demo_folder directory is not present then create it.
	    os.makedirs(target_dir)


def change_ext(renamee, new_extension):
    pre, ext = os.path.splitext(renamee)
    return os.rename(renamee, pre + new_extension)