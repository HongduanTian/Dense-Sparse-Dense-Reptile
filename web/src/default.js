var DEFAULT_PATHS = [[[[0.695,0.35],[0.695,0.36],[0.695,0.36],[0.695,0.36],[0.695,0.37],[0.695,0.37],[0.695,0.38],[0.695,0.38],[0.685,0.39],[0.685,0.39],[0.685,0.4],[0.685,0.41],[0.675,0.41],[0.675,0.41],[0.675,0.42],[0.675,0.43],[0.665,0.43],[0.665,0.43],[0.655,0.44],[0.655,0.45],[0.645,0.46],[0.645,0.47],[0.635,0.47],[0.635,0.48],[0.625,0.49],[0.625,0.5],[0.615,0.51],[0.615,0.52],[0.605,0.53],[0.605,0.53],[0.595,0.54],[0.595,0.54],[0.595,0.55],[0.585,0.56],[0.585,0.56],[0.575,0.57],[0.575,0.57],[0.575,0.58],[0.575,0.58],[0.565,0.59],[0.565,0.59],[0.565,0.59],[0.555,0.6],[0.555,0.6],[0.555,0.61],[0.555,0.61],[0.545,0.61],[0.545,0.62],[0.545,0.62],[0.545,0.62],[0.535,0.63],[0.535,0.63],[0.535,0.63],[0.535,0.63],[0.525,0.64],[0.525,0.64],[0.525,0.64],[0.525,0.64],[0.525,0.65],[0.515,0.65],[0.515,0.65],[0.515,0.65],[0.515,0.65],[0.505,0.65],[0.505,0.65],[0.505,0.65],[0.505,0.65],[0.495,0.65],[0.495,0.65],[0.495,0.65],[0.485,0.65],[0.475,0.65],[0.475,0.65],[0.465,0.65],[0.465,0.65],[0.455,0.65],[0.445,0.65],[0.445,0.65],[0.435,0.65],[0.425,0.65],[0.425,0.65],[0.415,0.65],[0.415,0.65],[0.405,0.65],[0.405,0.65],[0.395,0.64],[0.395,0.64],[0.385,0.64],[0.385,0.64],[0.385,0.64],[0.375,0.63],[0.375,0.63],[0.375,0.63],[0.375,0.63],[0.365,0.63],[0.365,0.62],[0.365,0.62],[0.365,0.61],[0.365,0.61],[0.355,0.61],[0.355,0.6],[0.355,0.6],[0.345,0.6],[0.345,0.59],[0.345,0.59],[0.345,0.58],[0.335,0.57],[0.335,0.56],[0.325,0.55],[0.325,0.54],[0.315,0.54],[0.315,0.53],[0.315,0.52],[0.305,0.51],[0.305,0.5],[0.305,0.5],[0.305,0.49],[0.305,0.49],[0.305,0.48],[0.295,0.47],[0.295,0.47],[0.295,0.46],[0.295,0.46],[0.295,0.45],[0.295,0.45],[0.295,0.44],[0.295,0.44],[0.295,0.44],[0.295,0.43],[0.295,0.43],[0.295,0.42],[0.295,0.42],[0.305,0.42],[0.305,0.42],[0.305,0.42],[0.305,0.41],[0.315,0.41],[0.315,0.41],[0.315,0.4],[0.315,0.4],[0.325,0.4],[0.325,0.4],[0.325,0.4],[0.325,0.4],[0.335,0.39],[0.335,0.39],[0.335,0.39],[0.335,0.39],[0.345,0.39],[0.345,0.39],[0.355,0.39],[0.355,0.39],[0.365,0.38],[0.365,0.38],[0.375,0.38],[0.375,0.38],[0.385,0.38],[0.395,0.38],[0.395,0.38],[0.405,0.37],[0.415,0.37],[0.425,0.37],[0.425,0.37],[0.435,0.37],[0.445,0.37],[0.445,0.37],[0.455,0.37],[0.465,0.37],[0.475,0.37],[0.475,0.37],[0.485,0.37],[0.485,0.37],[0.495,0.37],[0.495,0.37],[0.495,0.37],[0.505,0.37],[0.505,0.37],[0.505,0.37],[0.515,0.37],[0.515,0.37],[0.525,0.37],[0.525,0.38],[0.525,0.38],[0.535,0.38],[0.535,0.38],[0.535,0.39],[0.535,0.39],[0.545,0.39],[0.545,0.39],[0.545,0.4],[0.555,0.4],[0.555,0.4],[0.555,0.4],[0.565,0.41],[0.565,0.41],[0.565,0.42],[0.565,0.42],[0.575,0.42],[0.575,0.43],[0.575,0.43],[0.575,0.44],[0.585,0.44],[0.585,0.45],[0.585,0.45],[0.585,0.45],[0.585,0.46],[0.585,0.46],[0.595,0.46],[0.595,0.47],[0.595,0.47],[0.595,0.48],[0.595,0.48],[0.595,0.49],[0.595,0.49],[0.595,0.49],[0.605,0.5],[0.605,0.5],[0.605,0.51],[0.605,0.51],[0.605,0.51],[0.615,0.52],[0.615,0.52],[0.615,0.53],[0.615,0.53],[0.615,0.53],[0.625,0.54],[0.625,0.54],[0.625,0.54],[0.635,0.55],[0.635,0.55],[0.635,0.56],[0.635,0.56],[0.645,0.57],[0.645,0.57],[0.645,0.57],[0.645,0.58],[0.655,0.58],[0.655,0.58],[0.655,0.59],[0.655,0.59],[0.655,0.59],[0.665,0.6],[0.665,0.6],[0.665,0.6],[0.665,0.61],[0.675,0.61],[0.675,0.61],[0.675,0.61],[0.675,0.62]]],[[[0.235,0.35],[0.235,0.35],[0.235,0.35],[0.235,0.36],[0.235,0.37],[0.235,0.37],[0.235,0.37],[0.235,0.38],[0.235,0.39],[0.235,0.4],[0.235,0.41],[0.235,0.42],[0.235,0.43],[0.235,0.44],[0.235,0.45],[0.235,0.45],[0.235,0.46],[0.235,0.47],[0.235,0.48],[0.235,0.48],[0.235,0.49],[0.235,0.5],[0.235,0.5],[0.235,0.51],[0.235,0.51],[0.235,0.52],[0.235,0.53],[0.235,0.53],[0.235,0.54],[0.235,0.55],[0.235,0.55],[0.235,0.56],[0.235,0.57],[0.235,0.57],[0.235,0.58],[0.235,0.58],[0.235,0.59],[0.235,0.59],[0.235,0.6],[0.235,0.6],[0.235,0.61],[0.235,0.61],[0.235,0.62],[0.235,0.63],[0.235,0.63],[0.235,0.64],[0.235,0.64],[0.235,0.65],[0.235,0.65],[0.235,0.65],[0.235,0.65],[0.235,0.66],[0.235,0.66],[0.235,0.66],[0.235,0.66],[0.235,0.65],[0.235,0.65],[0.235,0.65],[0.235,0.64],[0.235,0.64],[0.235,0.63],[0.235,0.63],[0.235,0.63],[0.245,0.62],[0.245,0.62],[0.245,0.61],[0.245,0.61],[0.245,0.6],[0.255,0.59],[0.255,0.59],[0.255,0.58],[0.265,0.58],[0.265,0.57],[0.265,0.57],[0.265,0.56],[0.275,0.55],[0.275,0.55],[0.285,0.55],[0.285,0.54],[0.285,0.54],[0.295,0.53],[0.295,0.53],[0.295,0.53],[0.305,0.52],[0.315,0.51],[0.325,0.51],[0.335,0.5],[0.345,0.49],[0.355,0.48],[0.365,0.47],[0.385,0.46],[0.395,0.45],[0.405,0.45],[0.415,0.43],[0.425,0.43],[0.435,0.42],[0.445,0.41],[0.445,0.41],[0.455,0.4],[0.465,0.4],[0.465,0.39],[0.475,0.39],[0.475,0.39],[0.485,0.38],[0.485,0.38],[0.495,0.37],[0.505,0.37],[0.505,0.37],[0.515,0.37],[0.525,0.36],[0.535,0.36],[0.545,0.36],[0.555,0.35],[0.565,0.35],[0.575,0.35],[0.585,0.35],[0.585,0.35],[0.595,0.35],[0.595,0.35],[0.605,0.35],[0.605,0.35],[0.615,0.35],[0.615,0.35],[0.625,0.35],[0.625,0.35],[0.625,0.35],[0.635,0.35],[0.635,0.35],[0.635,0.35],[0.645,0.35],[0.645,0.35],[0.645,0.35],[0.645,0.35],[0.655,0.35],[0.655,0.35],[0.665,0.36],[0.665,0.36],[0.665,0.36],[0.665,0.36],[0.675,0.37],[0.675,0.37],[0.675,0.37],[0.675,0.37],[0.685,0.38],[0.685,0.38],[0.685,0.38],[0.695,0.39],[0.695,0.4],[0.695,0.4],[0.705,0.41],[0.705,0.41],[0.705,0.41],[0.715,0.42],[0.715,0.42],[0.715,0.43],[0.715,0.43],[0.715,0.43],[0.715,0.44],[0.715,0.44],[0.715,0.44],[0.715,0.45],[0.715,0.45],[0.715,0.46],[0.715,0.46],[0.715,0.47],[0.715,0.47],[0.715,0.48],[0.715,0.48],[0.715,0.49],[0.715,0.5],[0.715,0.5],[0.715,0.51],[0.715,0.52],[0.715,0.52],[0.715,0.53],[0.715,0.54],[0.715,0.54],[0.715,0.55],[0.715,0.56],[0.715,0.56],[0.705,0.57],[0.705,0.58],[0.705,0.59],[0.695,0.6],[0.695,0.6],[0.695,0.6],[0.695,0.61],[0.695,0.61],[0.695,0.61],[0.685,0.62],[0.685,0.62],[0.685,0.62],[0.685,0.62],[0.675,0.62],[0.675,0.62],[0.675,0.62],[0.675,0.62],[0.665,0.62],[0.665,0.62],[0.665,0.62],[0.655,0.62],[0.655,0.62],[0.655,0.62],[0.655,0.63],[0.645,0.63],[0.645,0.63],[0.645,0.63],[0.645,0.63],[0.635,0.63],[0.635,0.63],[0.635,0.63],[0.635,0.63],[0.625,0.63],[0.625,0.63],[0.625,0.63],[0.625,0.62],[0.615,0.62],[0.615,0.62],[0.615,0.62],[0.615,0.62],[0.605,0.62],[0.605,0.61],[0.595,0.61],[0.595,0.61],[0.585,0.61],[0.585,0.61],[0.575,0.6],[0.575,0.6],[0.565,0.6],[0.565,0.59],[0.555,0.59],[0.555,0.59],[0.545,0.58],[0.545,0.58],[0.535,0.58],[0.535,0.57],[0.525,0.57],[0.525,0.57],[0.515,0.56],[0.505,0.56],[0.495,0.55],[0.485,0.54],[0.475,0.53],[0.475,0.53],[0.465,0.52],[0.455,0.52],[0.445,0.51],[0.445,0.51],[0.435,0.51],[0.435,0.5],[0.435,0.5],[0.435,0.5],[0.425,0.5],[0.425,0.5],[0.425,0.5],[0.415,0.49],[0.415,0.49],[0.415,0.49],[0.405,0.49],[0.395,0.48],[0.395,0.48],[0.385,0.47],[0.375,0.47],[0.375,0.47],[0.365,0.46],[0.365,0.46],[0.355,0.46],[0.355,0.45],[0.345,0.45],[0.345,0.45],[0.335,0.44],[0.335,0.44],[0.325,0.43],[0.315,0.43],[0.315,0.42],[0.305,0.42],[0.305,0.42],[0.305,0.41],[0.295,0.41],[0.295,0.41],[0.295,0.41],[0.295,0.4],[0.285,0.4],[0.285,0.4],[0.285,0.4],[0.285,0.4],[0.285,0.4],[0.285,0.39],[0.285,0.39],[0.275,0.39],[0.275,0.39],[0.275,0.39],[0.275,0.39],[0.275,0.39],[0.275,0.38],[0.275,0.38],[0.275,0.38],[0.265,0.38],[0.265,0.38],[0.265,0.38],[0.265,0.38],[0.265,0.37],[0.265,0.37],[0.265,0.37],[0.265,0.37],[0.265,0.37],[0.265,0.36],[0.265,0.36],[0.255,0.36],[0.255,0.36],[0.255,0.35],[0.255,0.35]],[[0.625,0.47]]],[[[0.255,0.68],[0.265,0.68],[0.265,0.68],[0.275,0.68],[0.285,0.68],[0.295,0.68],[0.305,0.68],[0.325,0.68],[0.335,0.68],[0.355,0.68],[0.375,0.68],[0.395,0.68],[0.405,0.67],[0.425,0.67],[0.445,0.67],[0.455,0.67],[0.475,0.66],[0.485,0.66],[0.495,0.66],[0.515,0.66],[0.525,0.66],[0.535,0.66],[0.555,0.66],[0.555,0.66],[0.565,0.66],[0.575,0.66],[0.585,0.66],[0.595,0.66],[0.605,0.66],[0.615,0.66],[0.625,0.66],[0.635,0.66],[0.635,0.66],[0.645,0.66],[0.655,0.66],[0.675,0.66],[0.685,0.66],[0.695,0.66],[0.705,0.66],[0.705,0.66],[0.715,0.66],[0.725,0.66],[0.725,0.66],[0.725,0.66],[0.735,0.66],[0.735,0.66],[0.735,0.66],[0.735,0.66],[0.735,0.66],[0.725,0.66],[0.725,0.65],[0.725,0.65],[0.715,0.64],[0.715,0.64],[0.705,0.63],[0.695,0.62],[0.695,0.61],[0.685,0.61],[0.685,0.6],[0.675,0.6],[0.675,0.59],[0.665,0.59],[0.665,0.58],[0.655,0.57],[0.655,0.57],[0.645,0.57],[0.645,0.56],[0.645,0.55],[0.635,0.55],[0.635,0.55],[0.635,0.54],[0.625,0.53],[0.625,0.53],[0.625,0.53],[0.615,0.52],[0.615,0.51],[0.605,0.51],[0.605,0.5],[0.605,0.49],[0.595,0.48],[0.585,0.48],[0.585,0.47],[0.585,0.46],[0.575,0.45],[0.575,0.44],[0.565,0.44],[0.565,0.43],[0.565,0.42],[0.565,0.42],[0.565,0.42],[0.555,0.41],[0.555,0.41],[0.555,0.41],[0.555,0.41],[0.545,0.4],[0.545,0.4],[0.545,0.39],[0.545,0.39],[0.545,0.39],[0.535,0.38],[0.535,0.37],[0.535,0.37],[0.535,0.36],[0.525,0.35],[0.525,0.35],[0.525,0.34],[0.515,0.33],[0.515,0.33],[0.515,0.32],[0.505,0.31],[0.505,0.3],[0.495,0.29],[0.495,0.28],[0.495,0.28],[0.495,0.28],[0.495,0.27],[0.485,0.27],[0.485,0.27],[0.485,0.26],[0.485,0.26],[0.485,0.26],[0.485,0.26],[0.485,0.27],[0.485,0.27],[0.485,0.27],[0.485,0.27],[0.485,0.28],[0.485,0.28],[0.475,0.28],[0.475,0.28],[0.475,0.29],[0.475,0.29],[0.475,0.29],[0.475,0.29],[0.475,0.29],[0.475,0.3],[0.475,0.3],[0.475,0.3],[0.465,0.31],[0.465,0.31],[0.465,0.32],[0.455,0.33],[0.455,0.33],[0.445,0.34],[0.435,0.35],[0.435,0.36],[0.425,0.37],[0.425,0.38],[0.415,0.38],[0.405,0.39],[0.405,0.4],[0.395,0.4],[0.395,0.41],[0.385,0.42],[0.385,0.42],[0.375,0.43],[0.375,0.43],[0.375,0.43],[0.375,0.43],[0.375,0.44],[0.375,0.44],[0.365,0.44],[0.365,0.44],[0.365,0.44],[0.365,0.44],[0.365,0.45],[0.365,0.45],[0.355,0.46],[0.355,0.46],[0.345,0.47],[0.345,0.48],[0.335,0.49],[0.335,0.5],[0.325,0.5],[0.315,0.52],[0.315,0.53],[0.305,0.53],[0.305,0.54],[0.295,0.55],[0.295,0.55],[0.295,0.55],[0.295,0.56],[0.295,0.56],[0.295,0.56],[0.295,0.56],[0.295,0.56],[0.295,0.57],[0.295,0.57],[0.295,0.57],[0.295,0.57],[0.285,0.58],[0.285,0.59],[0.285,0.59],[0.275,0.6],[0.275,0.6],[0.265,0.61],[0.265,0.62],[0.265,0.62],[0.265,0.63],[0.265,0.63],[0.255,0.63],[0.255,0.63],[0.255,0.64],[0.255,0.64],[0.255,0.64],[0.255,0.64],[0.255,0.65]]],[[[0.285,0.45],[0.285,0.45],[0.285,0.45],[0.285,0.46],[0.285,0.46],[0.285,0.46],[0.285,0.47],[0.285,0.47],[0.285,0.47],[0.285,0.48],[0.285,0.48],[0.285,0.48],[0.285,0.48],[0.285,0.49],[0.285,0.49],[0.285,0.49],[0.285,0.5],[0.285,0.5],[0.285,0.51],[0.285,0.51],[0.285,0.52],[0.285,0.52],[0.285,0.53],[0.285,0.53],[0.285,0.54],[0.285,0.54],[0.285,0.55],[0.285,0.55],[0.285,0.56],[0.285,0.56],[0.285,0.56],[0.285,0.57],[0.285,0.57],[0.285,0.58],[0.285,0.58],[0.285,0.58],[0.285,0.58],[0.285,0.59],[0.285,0.59],[0.285,0.59],[0.295,0.6],[0.295,0.6],[0.295,0.6],[0.295,0.61],[0.295,0.61],[0.295,0.62],[0.295,0.62],[0.295,0.62],[0.295,0.63],[0.295,0.63],[0.295,0.63],[0.305,0.64],[0.305,0.64],[0.305,0.64],[0.305,0.65],[0.305,0.65],[0.305,0.65],[0.305,0.65],[0.305,0.66],[0.305,0.66],[0.305,0.66],[0.305,0.66],[0.305,0.67],[0.305,0.67],[0.305,0.67],[0.305,0.66],[0.305,0.66],[0.305,0.66],[0.315,0.66],[0.315,0.65],[0.315,0.65],[0.315,0.65],[0.315,0.64],[0.315,0.64],[0.315,0.64],[0.315,0.64],[0.315,0.63],[0.325,0.63],[0.325,0.63],[0.325,0.63],[0.325,0.63],[0.325,0.62],[0.325,0.62],[0.335,0.62],[0.335,0.62],[0.335,0.62],[0.335,0.61],[0.335,0.61],[0.335,0.61],[0.345,0.61],[0.345,0.6],[0.345,0.6],[0.345,0.6],[0.345,0.6],[0.355,0.59],[0.355,0.59],[0.355,0.59],[0.355,0.58],[0.355,0.58],[0.355,0.58],[0.365,0.58],[0.365,0.58],[0.365,0.57],[0.365,0.57],[0.365,0.57],[0.375,0.57],[0.375,0.56],[0.375,0.56],[0.385,0.56],[0.385,0.55],[0.395,0.54],[0.395,0.54],[0.395,0.54],[0.405,0.53],[0.405,0.52],[0.415,0.52],[0.415,0.52],[0.425,0.51],[0.425,0.5],[0.435,0.5],[0.435,0.5],[0.445,0.49],[0.445,0.49],[0.445,0.48],[0.455,0.48],[0.455,0.47],[0.455,0.47],[0.465,0.47],[0.465,0.46],[0.475,0.46],[0.475,0.45],[0.485,0.45],[0.495,0.44],[0.495,0.44],[0.505,0.43],[0.515,0.42],[0.525,0.42],[0.535,0.41],[0.545,0.4],[0.545,0.4],[0.555,0.39],[0.565,0.39],[0.565,0.38],[0.575,0.38],[0.575,0.38],[0.585,0.37],[0.585,0.37],[0.595,0.37],[0.595,0.36],[0.605,0.36],[0.615,0.36],[0.615,0.35],[0.625,0.35],[0.635,0.34],[0.635,0.34],[0.645,0.33],[0.645,0.33],[0.655,0.33],[0.665,0.33],[0.665,0.32],[0.675,0.32],[0.685,0.32],[0.695,0.31],[0.695,0.31],[0.705,0.31],[0.715,0.31],[0.715,0.31],[0.725,0.31],[0.725,0.31],[0.735,0.31],[0.735,0.31],[0.745,0.31],[0.755,0.3],[0.755,0.3],[0.765,0.3],[0.765,0.3],[0.775,0.3],[0.775,0.3],[0.785,0.3],[0.785,0.3],[0.795,0.3],[0.795,0.3],[0.795,0.3],[0.795,0.3],[0.805,0.3],[0.805,0.3],[0.805,0.3],[0.805,0.3],[0.815,0.31],[0.815,0.31],[0.815,0.31],[0.815,0.31],[0.815,0.31],[0.815,0.32],[0.815,0.32],[0.825,0.32],[0.825,0.32],[0.825,0.33],[0.825,0.33],[0.825,0.33],[0.835,0.33],[0.835,0.34],[0.835,0.34],[0.835,0.34],[0.835,0.34],[0.835,0.35],[0.845,0.35],[0.845,0.36],[0.845,0.36],[0.855,0.37],[0.855,0.37],[0.855,0.38],[0.855,0.38],[0.855,0.39],[0.865,0.39],[0.865,0.4],[0.865,0.41],[0.865,0.41],[0.865,0.41],[0.865,0.42],[0.865,0.42],[0.875,0.43],[0.875,0.43],[0.875,0.44],[0.875,0.44],[0.875,0.45],[0.875,0.45],[0.875,0.46],[0.875,0.46],[0.875,0.46],[0.875,0.47],[0.875,0.47],[0.875,0.48],[0.875,0.48],[0.875,0.49],[0.865,0.49],[0.865,0.5],[0.865,0.51],[0.865,0.51],[0.855,0.52],[0.855,0.53],[0.855,0.54],[0.855,0.55],[0.845,0.56],[0.845,0.57],[0.845,0.57],[0.845,0.59],[0.835,0.6],[0.835,0.61],[0.835,0.62],[0.835,0.63],[0.825,0.63],[0.825,0.64],[0.825,0.65],[0.825,0.66],[0.815,0.66],[0.815,0.67],[0.815,0.67],[0.805,0.68],[0.805,0.68],[0.805,0.68],[0.805,0.69],[0.795,0.69],[0.795,0.7],[0.795,0.7],[0.785,0.71],[0.785,0.71],[0.785,0.71],[0.775,0.72],[0.775,0.72],[0.775,0.72],[0.765,0.73],[0.765,0.73],[0.765,0.74],[0.765,0.74],[0.755,0.74],[0.755,0.75],[0.745,0.75],[0.745,0.76],[0.745,0.76],[0.745,0.76],[0.735,0.76],[0.735,0.77],[0.735,0.77],[0.735,0.77],[0.725,0.77],[0.725,0.77],[0.725,0.77],[0.725,0.77],[0.725,0.76],[0.715,0.76],[0.715,0.76],[0.715,0.76],[0.715,0.75],[0.715,0.75],[0.715,0.75],[0.705,0.74],[0.705,0.73],[0.705,0.73],[0.695,0.72],[0.685,0.71],[0.685,0.7],[0.675,0.7],[0.675,0.69],[0.665,0.69],[0.655,0.68],[0.655,0.68],[0.645,0.67],[0.635,0.67],[0.635,0.66],[0.625,0.66],[0.625,0.65],[0.615,0.64],[0.605,0.64],[0.595,0.63],[0.595,0.63],[0.585,0.62],[0.575,0.62],[0.565,0.61],[0.555,0.61],[0.555,0.61],[0.545,0.61],[0.535,0.6],[0.535,0.6],[0.525,0.6],[0.525,0.59],[0.515,0.59],[0.515,0.59],[0.505,0.59],[0.505,0.59],[0.495,0.58],[0.495,0.58],[0.495,0.58],[0.485,0.58],[0.485,0.57],[0.475,0.57],[0.475,0.57],[0.465,0.57],[0.465,0.56],[0.455,0.56],[0.455,0.55],[0.445,0.55],[0.445,0.55],[0.435,0.54],[0.435,0.54],[0.425,0.53],[0.415,0.53],[0.415,0.53],[0.415,0.53],[0.405,0.53],[0.405,0.52],[0.395,0.52],[0.395,0.52],[0.395,0.52],[0.385,0.52],[0.385,0.51],[0.385,0.51],[0.375,0.51],[0.375,0.51],[0.365,0.5],[0.365,0.5],[0.355,0.5],[0.355,0.5],[0.355,0.5],[0.355,0.5],[0.345,0.5],[0.345,0.5],[0.345,0.5],[0.345,0.49],[0.335,0.49],[0.335,0.49],[0.335,0.49],[0.335,0.49],[0.335,0.49],[0.325,0.49],[0.325,0.49],[0.325,0.49],[0.325,0.48],[0.325,0.48],[0.315,0.48],[0.315,0.48],[0.315,0.48],[0.315,0.48],[0.315,0.48],[0.305,0.47],[0.305,0.47],[0.305,0.47],[0.305,0.47],[0.305,0.47],[0.295,0.47],[0.295,0.47],[0.295,0.47],[0.295,0.46],[0.295,0.46],[0.295,0.46],[0.295,0.46]],[[0.725,0.43]]]];

var DEFAULT_PROBS = [0.00040970481390258584, 0.9954833771966454, 0.004106918130318439];
