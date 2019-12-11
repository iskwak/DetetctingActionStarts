"""Helper functions to get git information."""
import os
import git


def log_git_status(filename):
    """Output the status of the git repository.

    Assumes that module is called from the subdirectory of the repository and
    that the origin url is 'git@github.com:kristinbranson/QuackNN'. Given
    an output filename, This function will log the current git commit id
    (hexsha) and whether or not the repository is dirty (additionally will
    provide minimal diff information.)
    """
    repo = get_repo()

    # get the origin url
    with open(filename, "w") as outfile:
        origin_url = get_origin_url(repo)

        if origin_url != 'git@github.com:kristinbranson/QuackNN' and\
           origin_url != 'git@github.com:kristinbranson/QuackNN.git':
            # if the origin url is not as expected, then set repo to None, and
            # let the other helper functions gracefully handle the None's.
            repo = None
            outfile.write("unknown repository\n")
        outfile.write(origin_url)
        outfile.write("\n")

        commit_id = get_commit_id(repo)
        outfile.write(commit_id)
        outfile.write("\n\n")

        is_dirty = is_repo_dirty(repo)

        if is_dirty is True:
            diffs = get_diffs(repo)
            outfile.write(diffs)
        else:
            outfile.write("clean repository\n")


def get_repo():
    """Get a repo object to query.

    This code assumes that it is called from either the repository root
    directory, or a subfolder of the repository root.

    If this function ends up being used a lot, chances are it would be easier
    to make a wrapper class for the commonly used git python operations. Until
    then, this will be used.
    """
    repo_dir = os.getcwd()
    git_dir = os.path.join(repo_dir, ".git")
    if os.path.isdir(git_dir) is True:
        repo = git.Repo(repo_dir)
    elif os.path.isdir(os.path.join(repo_dir, "..", ".git")) is True:
        repo = git.Repo(os.path.join(repo_dir, ".."))
    else:
        repo = None

    return repo


def get_commit_id(repo):
    """Get the most recent commit id of the repository."""
    if repo is None:
        commit_id = "unknown"
    else:
        commit_id = repo.head.commit.hexsha

    return commit_id


def is_repo_dirty(repo):
    """Get the most recent commit id of the repository."""
    if repo is None:
        is_dirty = None
    else:
        is_dirty = repo.is_dirty()

    return is_dirty


def get_diffs(repo):
    """Get any diffs for the repository."""
    if repo is None:
        diff_str = None
    else:
        git_diffs = repo.head.commit.diff(None)
        diff_str = ""
        for git_diff in git_diffs:
            diff_str += git_diff.__str__()
            # add a new line to the diff concatenation
            diff_str += "\n"

    return diff_str


def get_origin_url(repo):
    """Get the url for the origin repository."""
    if repo is None:
        origin_url = None
    else:
        origins = repo.remote('origin').urls
        # origin should only have one url... i think?
        origin_url = next(origins)

    return origin_url
