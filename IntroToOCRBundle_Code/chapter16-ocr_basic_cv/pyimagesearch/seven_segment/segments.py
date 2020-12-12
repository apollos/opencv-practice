# define a dictionary that maps each possible digit to which segments
# are "on"
DIGITS = {
	0: (1, 1, 1, 0, 1, 1, 1),
	1: (0, 0, 1, 0, 0, 1, 0),
	2: (1, 0, 1, 1, 1, 0, 1),
	3: (1, 0, 1, 1, 0, 1, 1),
	4: (0, 1, 1, 1, 0, 1, 0),
	5: (1, 1, 0, 1, 0, 1, 1),
	6: (1, 1, 0, 1, 1, 1, 1),
	7: (1, 0, 1, 0, 0, 1, 0),
	8: (1, 1, 1, 1, 1, 1, 1),
	9: (1, 1, 1, 1, 0, 1, 1),
}

# use the digits dictionary to define an inverse dictionary which maps
# segments that are turned on vs. off to their corresponding digits
DIGITS_INV = {v:k for (k, v) in DIGITS.items()}