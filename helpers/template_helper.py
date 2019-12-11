"""Helper to deal with the graph templates."""
import re


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
