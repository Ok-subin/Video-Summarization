import matplotlib.pyplot as plt
import numpy as np

def draws(titleName, num_frames, probs, machine_summary, cps, positions, n_steps):
    # picks/positions/15프레임 간격마다. 기본이 되는 회색 그래프
    x = np.arange(num_frames)
    values = []

    if positions.dtype != int:
        positions = positions.astype(np.int32)
    if positions[-1] != num_frames:
        positions = np.concatenate([positions, [num_frames]])
    for i in range(len(positions) - 1):
        for j in range(15):
            if (len(values) >= len(x)):
                break
            values.append(probs[i])       # 15 간격마다의 값을 구해서 values에 넣어줌

    colors = []
    
    for i in range(num_frames):
        if (machine_summary[i] == 1.0):
            colors.append('red')
        else:
            colors.append('grey')

    plt.bar(x, values, color=colors)
    title = "./" + str(titleName) + ".jpg"
    plt.savefig(title)