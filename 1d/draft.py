import numpy as np

# Create a sample complex array
complex_array = np.array([
    1.000000000000000000e+00 + 0.000000000000000000e+00j,
    9.999500024998958514e-01 - 9.999625015884732598e-03j,
    9.998000349959171862e-01 - 1.999725036505866152e-02j
])

# Save the complex array to a text file with a specific format
with open('complex_array.txt', 'w') as f:
    for number in complex_array:
        f.write(f"{number.real:.18e}{number.imag:+.18e}j\n")
# Load the complex array from the text file
def complex_converter(s):
    return complex(s.replace('j', '1j'))

loaded_array = np.loadtxt('complex_array.txt', dtype=complex, converters={0: complex_converter})

print(loaded_array)
