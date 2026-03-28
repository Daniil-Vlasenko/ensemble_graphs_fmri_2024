import nibabel
import hcp_utils
import numpy
import pandas
nibabel.imageglobals.logger.setLevel(40)


def parcellation(files_in, files_out):
    """
    Parcellation of the original fMRI files into regions, averaging the time series for each region and saving the
    time series of the regions.

    Parameters
    ----------
    files_in : list
        List of paths to source fMRI (CIFTI) data.
    files_out : list
        List of paths to save parcelled data.

    Returns
    -------
    None
    """
    for file_in, file_out in zip(files_in, files_out):
        img = nibabel.load(file_in)
        data = img.get_fdata()
        data_p = hcp_utils.parcellate(data, hcp_utils.mmp)
        data_pd = pandas.DataFrame(data_p.T, index=hcp_utils.mmp.nontrivial_ids)
        data_pd.to_csv(file_out, index=True)


def detrend(files_in, files_out):
    """
    Removing a linear trend of the time series of regions.

    Parameters
    ----------
    files_in : list
        List of paths to parcelled data.
    files_out : list
        List of paths to save the data without a linear trend

    Returns
    -------
    None
    """
    for file_in, file_out in zip(files_in, files_out):
        data = pandas.read_csv(file_in, index_col=0)

        for i in range(data.shape[0]):
            y = data.iloc[i, :].copy()
            x = [j for j in range(len(y))]

            coeffs = numpy.polyfit(x, y, 1)
            trend = numpy.polyval(coeffs, x)
            detrended_y = y - trend
            data.iloc[i, :] = detrended_y

        data.to_csv(file_out, index=True)


def normalize(files_in, files_out):
    """
    Normalization of the time series of regions.

    Parameters
    ----------
    files_in : list
        List of paths to parcelled data without a linear trend.
    files_out : list
        List of paths to save the normalized data.

    Returns
    -------
    None
    """
    for file_in, file_out in zip(files_in, files_out):
        data = pandas.read_csv(file_in, index_col=0)

        normalized_data = data.sub(data.mean(axis=1), axis=0)
        normalized_data = normalized_data.div(data.std(axis=1), axis=0)

        normalized_data.to_csv(file_out, index=True)


# Information about the number of fMRI volumes and duration of experiments from HCP.
time = {"wm": [405, 301],
        "gambling": [253, 192],
        "motor": [284, 214],
        "language": [316, 237],
        "social": [274, 207],
        "relational": [232, 176],
        "emotion": [176, 136]}


def splitting(files_in1, files_in2, files_out, task):
    """
    Splitting the data into two cognitive states according to the information from HCP EV files.
    There are two EV files for gambling, language, relational, emotion, social, eight EV files for wm, and four
    EV files for motor (we ignore the tongue movement).

    Parameters
    ----------
    files_in1 : list
        List of paths to the normalized data.
    files_in2 : list
        List of paths to EV files.
    files_out : list
        List of paths to save the split data.
    task : string
        Type of experiment (gambling, language, relational, emotion, social, wm, motor)

    Returns
    -------
    None
    """
    for i in range(len(files_in1)):
        if task in ["gambling", "language", "relational", "emotion", "social"]:
            EV_files = [files_in2[0][i], files_in2[1][i]]
            EV = [numpy.loadtxt(EV_files[0]), numpy.loadtxt(EV_files[1])]
        if task == "wm":
            EV_files = [files_in2[0][4 * i:4 * i + 4], files_in2[1][4 * i:4 * i + 4]]
            EV = [numpy.vstack((numpy.loadtxt(EV_files[0][0]), numpy.loadtxt(EV_files[0][1]),
                                numpy.loadtxt(EV_files[0][2]), numpy.loadtxt(EV_files[0][3]))),
                  numpy.vstack((numpy.loadtxt(EV_files[1][0]), numpy.loadtxt(EV_files[1][1]),
                                numpy.loadtxt(EV_files[1][2]), numpy.loadtxt(EV_files[1][3])))]
        if task == "motor":
            EV_files = [files_in2[0][2 * i:2 * i + 2], files_in2[1][2 * i:2 * i + 2]]
            EV = [numpy.vstack((numpy.loadtxt(EV_files[0][0]), numpy.loadtxt(EV_files[0][1]))),
                  numpy.vstack((numpy.loadtxt(EV_files[1][0]), numpy.loadtxt(EV_files[1][1])))]

        data = pandas.read_csv(files_in1[i], index_col=0)
        for k in range(len(EV)):
            for j in range(EV[k].shape[0]):
                frame1 = round(EV[k][j][0] / (time[task][1] / time[task][0]))
                frame2 = round((EV[k][j][0] + EV[k][j][1]) / (time[task][1] / time[task][0]))
                splitted_data = data.iloc[:, frame1:frame2]

                if j == 0:
                    merged_data = splitted_data
                else:
                    merged_data = pandas.concat([merged_data, splitted_data], axis=1)

            merged_data.to_csv(files_out[2 * i + k], index=True)
