import json, warnings, gc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
import argparse


def get_random_sentence(
    max_words: int = 6,
    min_words: int = 3,
    similar_word_map: dict = {},
    places: list = [],
    numericals: list = [],
):
    """
    Get a random title or axis label for each plot
    """
    
    # Get title length
    title_length = np.random.randint(min_words, max_words+1)
    
    if title_length == 1:
        return np.random.choice(similar_word_map[np.random.randint(0, len(similar_word_map))])
    
    # Select random words (can be duplicated)
    title = [np.random.choice(similar_word_map[i]) for i in np.random.choice(range(0, len(similar_word_map)), size=title_length-1)]
    
    # Select a place
    title[-1] = np.random.choice(places)
    
    # Shuffle final title
    np.random.shuffle(title)
    
    # Add numerical
    title.append(np.random.choice(numericals))
    
    return " ".join(title)[:50]

def sample_polynomial(polynomial_coeffs, x_min, x_max, num_samples):
    """Samples random values from a polynomial.

    Args:
        polynomial_coeffs (array-like): Coefficients of the polynomial from highest to lowest degree.
        x_min (float): Minimum x value to sample.
        x_max (float): Maximum x value to sample.
        num_samples (int): Number of samples to take.

    Returns:
        ndarray: A 2D array of shape (num_samples, 2) containing (x, y) pairs.
        
    Source: ChatGPT
    """
    # Create an array of x values to sample
    x = np.random.uniform(x_min, x_max, num_samples)

    # Evaluate the polynomial at the x values to get the y values
    y = np.polyval(polynomial_coeffs, x)
    return y


def get_numerical_series(
    poly_pct: float = 0.6,
    linear_pct: float = 0.2,
    part_numerical_pct: float = 0.2,
    series_size: int = 8,
    data_type: str = 'float',
    random_outlier_pct: float = 0.1,
    force_positive: bool = True,
    cat_type: str = "",
):
    # Array to return
    res = []
    
    # Multiplication factor
    mult_prob = np.random.choice([0.01, 0.1, 10, 100, 1000, 1000000], 1, p=[0.04, 0.15, 0.55, 0.15, 0.10, 0.01])[0]
    mf = np.random.normal(mult_prob)
    
    # Sample strategy
    if cat_type == "year":
        gap = np.random.choice([1,2,5,10])
        start = np.random.randint(1900, 2100 - (series_size*gap))
        res = [int(start + i*gap) for i in range(series_size)]
        return res
    else:
        cat_type = np.random.choice(["poly", "linear"], 1, p=[0.5, 0.5])[0]
        if cat_type == "poly":
            res = sample_polynomial(
                polynomial_coeffs = [np.random.randint(1, 5), np.random.randint(1, 5), np.random.randint(0, 5)], 
                x_min=-1, 
                x_max=1, 
                num_samples=series_size,
            )
        elif cat_type == "linear":
            # Note: Not completely linear, but the last coeff acts as noise
            res = sample_polynomial(
                polynomial_coeffs = [np.random.randint(1, 5), np.random.randint(1, 5), 1+np.random.random()/10], 
                x_min=-1, 
                x_max=1, 
                num_samples=series_size,
            )
    
    # Force Positive
    if force_positive:
        res = [x * -1 if x < 0 else x for x in res]
        
    # Add multiplication factor
    res = [x*mf for x in res]
    
    # Set data type
    if data_type == "float":
        res = [np.clip(np.round(x, 2), 0, 10_000_000) for x in res]
        res = [np.round(np.random.uniform(0.2, 1.0), 2) if x < 0.00000001 else x for x in res] # corrects values that are all 0
    else:
        res = [np.clip(int(x), 0, 10_000_000) for x in res]
        res = [np.random.randint(1, 20) if x==0 else x for x in res] # corrects values that are all 0

        
    # Apply random outlier
    if np.random.random() < random_outlier_pct:
        res[np.random.randint(0, len(res))] *= 2
        
    return res


def get_categorical_series(
    noun_pct: float = 0.60,
    place_pct: float = 0.30,
    part_numerical_pct: float = 0.1,
    series_size: int = 8,
    similar_word_map: dict = {},
    places: list = [],
):
    # Array to return
    res = []
    cat_type = np.random.choice(["noun", "place", "part_num"], 1, p=[noun_pct, place_pct, part_numerical_pct])[0]
    
    # Catches case when word is duplicated
    while True:

        # Select based on type
        if cat_type == "noun":
            vals = similar_word_map[np.random.randint(0, len(similar_word_map))]
            if len(vals) <= series_size:
                res = vals
            else:
                res = np.random.choice(vals, series_size, replace=False)
        elif cat_type == "place":
            res = np.random.choice(places, series_size, replace=True)
        elif cat_type == "part_num":
            res = get_numerical_series(
                series_size = series_size,
                data_type = "int"
            )
            # Add symbol before or after number
            symbol = np.random.choice(['%', '$', 'mb', 'id', 'gb', 'k', 'hz', 'kg', 'qt'])
            if np.random.random() > 0.5:
                res = ["{}{}".format(x, symbol) for x in res]
            else:
                res = ["{}{}".format(symbol, x) for x in res]
            pass
        else:
            raise ValueError("Invalid cat_type.")
        
        # Making sure strings are not too long
        res = [x[:50] for x in res]
        
        # Catches random case when word is repeated in the data
        if len(list(set(res))) == len(res):
            break
            
    return res


def plot_vert_bar(
    xs: list = [],
    ys: list = [], 
    series_size: int = 6,
    remove_spines: bool = False,
    remove_ticks: bool = False,
    bar_color: str = "tab:blue",
    style: str = "classic",
    font_family: str = "DejaVu Sans",
    font_size: int = 10,
    edge_color: str = "black",
    font_style: str = "normal",
    font_weight: str = "normal",
    title_pct: float = 0.5,
    axis_title_pct: float = 0.5,
    cat_type: str = "", # if set to year, formats labels
    fnum: int = 0,
):  
    
    
    # Set default style
    with plt.style.context(style):
        
        # Font params
        plt.rcParams['font.size'] = font_size
        plt.rcParams["font.family"] = font_family
        plt.rcParams["font.style"] = font_style
        plt.rcParams['font.weight'] = font_weight
        
        # Create figure
        fig, ax = plt.subplots(figsize=(6.5, 4.5), num=1, clear=True)

        if remove_spines == True:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

        if remove_ticks == True:
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')

        # Add variation in the margins
        plt.margins(x=np.random.uniform(0.0, 0.15), y=0.1)

        # Display the plot
        if type(xs[0]) == str:
            ax.bar(xs, ys, color=bar_color, width=np.random.uniform(0.5, 0.8), zorder=3, edgecolor=edge_color)
        else:
            ax.bar(xs, ys, color=bar_color, width=((max(xs) - min(xs))/len(xs))*np.random.uniform(0.75,1), zorder=3, edgecolor=edge_color)

        # WORDS Only: Rotate categorical labels when they likely overlap
        if type(xs[0]) == str and (series_size >= 8 or sum([len(x) for x in xs]) > 40 or max([len(x) for x in xs]) > 9):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=np.random.choice([45,67.5,90]))

        # YEAR Only: Forces x-axis year labels as integers 
        if cat_type == "year":
            labels = []
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if int(label._x) != float(label.get_text()):
                    label.set_visible(False)
                else:
                    label._text = label.get_text().split('.')[0]
                labels.append(label)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax.set_xticklabels(labels)

        # Add Titles
        if np.random.random() < title_pct:
            title = get_random_sentence(
                max_words=6,
                min_words=4,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.title(title, pad=np.random.randint(0, 20))

        if np.random.random() < axis_title_pct:
            # Set X axis label
            xtitle = get_random_sentence(
                max_words=4,
                min_words=1,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.xlabel(xtitle)

            # Set Y Axis Label
            ytitle = get_random_sentence(
                max_words=4,
                min_words=1,
                similar_word_map=similar_word_map,
                places=places,
                numericals=numericals,
            )   
            plt.ylabel(ytitle)

    #         # Plot width x weight
    #         fig_width, fig_height = plt.gcf().get_size_inches()*fig.dpi
    #         print(fig_width, fig_height)

        # Show plot
        fig.savefig(config.out_file +'/{}.jpg'.format(fnum), bbox_inches='tight')

        plt.close("all")
        plt.close()
        gc.collect()

    # Reset default font size
    plt.rcParams['font.size'] = 10

def create_vertical_bars(n):
    """
    Creates N vertical bar charts.
    """
    
    meta_data = {}
    for i in tqdm(range(n)):

        # Sample from styles
        rspines = np.random.choice(remove_spines)
        rticks = np.random.choice(remove_ticks)
        bcolor = np.random.choice(bar_colors)
        mplstyle = np.random.choice(mpl_styles)
        ffamily = np.random.choice(font_families)
        fsize = np.random.choice(font_sizes)
        ecolor = np.random.choice(edge_colors)
        fstyle = np.random.choice(font_styles)
        fweight  = np.random.choice(font_weights)
        ss = np.random.choice(series_sizes)

        # Sample data series type based on probability from calculation in Frequencies section 
        x_type = np.random.choice(x_counts["vertical_bar"]['dst'], p=x_counts["vertical_bar"]['pct'])
        y_type = np.random.choice(y_counts["vertical_bar"]['dst'], p=y_counts["vertical_bar"]['pct'])

    #     # Log Style
    #     print((rspines, rticks, bcolor, mplstyle, ffamily, fsize, ecolor, fstyle, fweight))

        # Categorical data series
        if x_type == 'str':
            cat_type = ""
            xs = get_categorical_series(
                series_size = ss,
                places = places,
                similar_word_map = similar_word_map,
            )   
        # Year data series (as int)
        else:
            cat_type = "year" # set cat_type for x_tick label fix
            xs = get_numerical_series(
                series_size = ss,
                data_type = np.random.choice(['int', 'float']),
                cat_type = "year",
            )

        ys = get_numerical_series(
            series_size = ss,
            data_type = y_type,
        )
        
        # Store MetaData
        x_series = ";".join([str(x) for x in xs])
        y_series = ";".join([str(y) for y in ys])
        meta_data[i] = {'xs': x_series, 'ys': y_series}

        # Create bar chart
        plot_vert_bar(
            xs = xs,
            ys = ys,
            series_size = ss,
            remove_spines = rspines,
            remove_ticks = rticks,
            bar_color = bcolor,
            style = mplstyle,
            font_family = ffamily,
            font_size = fsize,
            edge_color = ecolor,
            font_style = fstyle,
            font_weight = fweight,
            cat_type = cat_type,
            fnum = i,
        )
        
    # Close last img
    plt.close()
        
    # Write MetaData to Disk
    with open(config.out_file + '/metadata.json', 'w') as f:
        json.dump(meta_data, f)

def parse_args():
    # Default values
    default_config = SimpleNamespace(
        out_file="./graphs",
        generate_n_imgs=10,
        seed=0,
        categoricals_file="./categoricals.json",
    )
    # Argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out_file", type=str, default=default_config.out_file, help="Directory to save charts in.")
    parser.add_argument("--generate_n_imgs", type=int, default=default_config.generate_n_imgs, help="Number of charts to generate.")
    parser.add_argument("--seed", type=int, default=default_config.seed, help="Seed used to generate data.")
    parser.add_argument("--categoricals_file", type=str, default=default_config.categoricals_file, help="File location of the categoricals.json file (w/ categorical series).")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get config
    config = parse_args()

    x_counts = {'dot': {'dst': ['int', 'str'], 'pct': [0.26, 0.74]}, 'line': {'dst': ['int', 'str'], 'pct': [0.37, 0.63]}, 'scatter': {'dst': ['float'], 'pct': [0.99]}, 'vertical_bar': {'dst': ['int', 'str'], 'pct': [0.3, 0.7]}, 'horizontal_bar': {'dst': ['int', 'float'], 'pct': [0.16, 0.84]}}
    y_counts = {'dot': {'dst': ['int'], 'pct': [1.0]}, 'line': {'dst': ['int', 'float'], 'pct': [0.12, 0.88]}, 'scatter': {'dst': ['int', 'float'], 'pct': [0.13, 0.87]}, 'vertical_bar': {'dst': ['int', 'float'], 'pct': [0.16, 0.84]}, 'horizontal_bar': {'dst': ['int', 'str'], 'pct': [0.3, 0.7]}}

    numericals = ['boeremag', 'soleil', 'rpms', 'kiloliters', 'vitamin', 'galerie', 'wastewater treatment', 'before', 'max', 'glabrous', 'subzero', 'consumes', 'per capita waste generation', 'revenue', 'near', 'pages', 'dissolved oxygen concentration', 'raging', 'torrential', 'inch', 'east', 'fraport', 'depending', 'profit margin', 'cylinder', 'materials', 'megalitres', 'renewable energy consumption', 'age', 'management', 'market share', 'millilitre', 'phase', 'renewable water resources', 'passing', 'taxes', 'karry', 'dynes', 'boltzmann', 'corpuscular', 'downpours', 'plan', 'exported', 'audible', 'fertility rate', 'equal', 'stress', 'wyville', 'digestibility score', 'cubic inches', 'buying', 'butter', 'henriksen', 'grain size', 'water efficiency', 'kilovolt', 'water content', 'tax rate', 'ardh', 'frames per second', 'prices', 'bentley', 'speed', 'engulfed', 'kiloliter', 'total fat content', 'expenditure', 'weighing', 'azria', 'infinitesimals', 'ohms', 'measuring', 'surface', 'sea level rise', 'heat', 'frequencies', 'stac', 'zero', 'turbo', 'stockpiles', 'forest cover', 'then', 'boosted', 'texture profile analysis', 'dividends', 'fice', 'kash', 'phytic acid content', 'singa', 'thomas', 'devastating', 'heart rate', 'mixing', 'powerbook', 'nease', 'kilometres', 'altimetry', 'score', 'intensity', 'powder', 'terrestrial protected areas coverage', 'threw', 'deferred', 'watt', 'albers', 'half', 'quarterly', 'aira', 'cards', 'hugh', 'kilojoules', 'bowls', 'harsiese', 'present', 'krw', 'tesla', 'refers', 'cubic meters', 'jobless', 'scorching', 'pint', 'mileage', 'metres', 'wildfire', 'direction', 'rebates', 'water solubility index', 'loan', 'square feet', 'clerc', 'diable', 'market', 'valuation', 'investment', 'speeds', 'loads', 'already', 'cuts', 'hz', 'pretax', 'relative humidity', 'mtoe', 'raisins', 'hayes', 'chatta', 'PHP', 'southwest', 'scd', 'kilobyte', 'carl', 'total solids content', 'densities', 'ridership', 'since', 'profit', 'ratio', 'coriander', 'minimum wage', 'bushfires', 'periods', 'investments', 'millimetres', 'shurik', 'pesticide regulation stringency', 'powerpc', 'bosomy', 'child', 'mhz', 'imported', 'mazda', 'ptf', 'water activity', 'snowfalls', 'bowl', 'tall', 'cholesterol content', 'hunger rate', 'sigma', 'read', 'grobler', 'liouville', 'outside', 'bmus', 'unsold', 'cpu', 'plastic waste generation', 'euler', 'handfuls', 'square inches', 'kramer', 'proportion', 'consumer', 'collection', 'manufacturing', 'earlier', 'kilobytes', 'meter', 'applies', 'pay', 'excluding', 'cup', 'spewed', 'moreover', 'fee', 'altitude', 'freshwater scarcity index', 'ounces', 'milliwatts', 'shf', 'once', 'coordinates', 'generating', 'bucks', 'overdrive', 'bailout', 'intervals', 'global warming potential', 'schenk', 'tailpipe', 'protein content', 'wildfires', 'calories', 'farenheit', 'petabytes', 'saucepan', 'catches', 'nanogram', 'lenders', 'account', 'total assets', 'rocknroll', 'harrison', 'exceeded', 'spent', 'expenses', 'damphousse', 'thread', 'seismicity', 'gdp', 'during', 'products', 'granulated', 'kilometers', 'infrasound', 'price', 'honeywell', 'angular', 'buk', 'environmental governance index', 'distances', 'in', 'global temperature', 'begin', 'purchases', 'ventresca', 'portfolio', 'clean energy production', 'sustainable agriculture policy stringency', 'but', 'loi', 'manning', 'cendant', 'assets', 'retail', 'square centimeters', 'picofarad', 'waist', 'dividend', 'wind energy production', 'uncompressed', 'fires', 'shrank', 'values', 'capella', 'sodium', 'simmer', 'fees', 'salaries', 'THB', 'turbines', 'nomi', 'rainy', 'income', 'billion', 'polyphenol content', 'turbofans', 'disk', 'mile', 'derived', 'caceres', 'articles', 'bpu', 'business', 'AUD', 'threatened species protection', 'toys', 'water availability', 'minimal', 'thermal conductivity', 'rtl', 'property value', 'vitamin c content', 'banking', 'protected area coverage', 'passes', 'fiscal year', 'gross energy value', 'rated', 'voicestream', 'microfarad', 'start', 'water stress', 'kilometers per hour', 'creuset', 'radioactive pollution', 'nini', 'bbb', 'value', 'village', 'johnny', 'brix value', 'wappingers', 'mmhg', 'maison', 'kilocycles', 'quarts', 'grammes', 'mccs', 'dyne', 'regions', 'blazes', 'credit score', 'profits', 'pensions', 'cubes', 'difference', 'balance', 'typical', 'radians', 'grams', 'km', 'continuous', 'time', 'meyer', 'debts', 'bruce', 'funds', 'inventory', 'humidity', 'increases', 'vertical', 'thermal', 'pectin content', 'mapped', 'fares', 'funding', 'petabyte', 'keystrokes', 'megawatt', 'missed', 'gigabits', 'town', 'XRP', 'humidity level', 'yearly income', 'energy density', 'cash flow', 'barrels', 'birthrates', 'cinnamon', 'ludwig', 'kia', 'mm', 'letters', 'barometric', 'toxicity index', 'degress', 'spending', 'temperatures', 'rushing', 'severance', 'iron content', 'horizontal', 'stories', 'grit size', 'sulfite content', 'cavus', 'from', 'been', 'inches', 'CAD', 'metric', 'dama', 'swept', 'kilowatt', 'levins', 'rambow', 'garlic', 'rate', 'pentium', 'skim', 'chevrolet', 'days', 'expenditures', 'water consumption per capita', 'yeast and mold count', 'starts', 'gross', 'medicare', 'hydrogenation level', 'purchasing', 'change', 'less', 'kvs', 'flood risk', 'torque', 'coastal habitat quality', 'reduced', 'food waste generation', 'seconds', 'storage', 'holway', 'dabhol', 'size', 'square miles', 'cialdini', 'registered', 'montano', 'milliohm', 'lines per inch', 'frederick', 'pixels', 'repayment', 'particular', 'birgeneau', 'microgrammes', 'body mass index', 'kilopascals', 'rechargeable', 'sak', 'week', 'wetland coverage', 'diems', 'hpa', 'varies', 'payouts', 'only', 'salt', 'fraternity', 'radius', 'lumix', 'bottles', 'wage', 'share', 'vogel', 'declines', 'lived', 'floods', 'floppies', 'downgrade', 'lipid oxidation', 'fall', 'schiavelli', 'offset', 'metre', 'ecliptic', 'mental health score', 'usage', 'birthrate', 'repay', 'dreier', 'output', 'microsecond', 'microgram', 'benthic habitat quality', 'gigabyte', 'exceeds', 'mbps', 'horsepower', 'jewelry', 'weights', 'hours', 'fumarate', 'due', 'fatty acid profile', 'nauk', 's', 'batteries', 'pfn', 'undergraduate', 'hovered', 'copy', 'SGD', 'inhabitants', 'touchscreen', 'KRW', 'tablespoons', 'lowest', 'broth', 'stein', 'body temperature', 'CHF', 'westinghouse', 'disks', 'abdulhakim', 'coral reef health index', 'wholesale', 'pesticide use', 'web', 'avis', 'fat', 'height', 'cruze', 'supply', 'leibniz', 'gramme', 'kgs', 'nantha', 'this', 'projected', 'ferrars', 'species richness', 'borrowing', 'blood pressure', 'reliability', 'west', 'khz', 'girth', 'short', 'kb', 'printouts', 'nearest', 'benefits', 'marine ecosystem health index', 'achieve', 'frequency', 'wielder', 'compensation', 'doppler', 'nanometers', 'ampere', 'epsilon', 'trade balance', 'miles per hour', 'psi', 'ifr', 'karmazin', 'electricity', 'trehalose', 'centimetres', 'along', 'payroll', 'franz', 'residual sugar level', 'accumulations', 'cholesterol', 'roi', 'furthermore', 'spilled', 'belletti', 'volt', 'carbohydrate', 'grayscale', 'mg', 'patient satisfaction', 'greenhouse gas emissions', 'charger', 'ericson', 'weighs', 'phosphorus balance', 'mth', 'fumble', 'ago', 'weierstrass', 'paid', 'watts', 'micrometres', 'text', 'lambda', 'proposing', 'affected', 'metamaterials', 'ravn', 'differs', 'water consumption', 'angle', 'kartheiser', 'cash', 'raged', 'amino acid score', 'length', 'heroin', 'monopulse', 'areas', 'twh', 'ratios', 'gunther', 'land area', 'terabytes', 'ended', 'lifespan', 'raise', 'adolphe', 'microliter', 'food', 'sips', 'aquifer depletion rate', 'standard', 'rainfalls', 'kilohertz', 'servings', 'chemical pollution index', 'bottle', 'ilm', 'recently', 'miles', 'long', 'ozone depletion potential', 'geographical', 'octets', 'confidence', 'opl', 'normal', 'tackles', 'illustrated', 'package', 'fahrenheit', 'retirement', 'boiling', 'kulasegaran', 'needed', 'essays', 'keeping', 'tonnes', 'comparison', 'months', 'repayments', 'around', 'minutes', 'editions', 'medroxyprogesterone', 'gibson', 'each', 'inflation rate', 'theorem', 'HKD', 'lamborghini', 'living', 'roentgens', 'curvature', 'nitrogen oxide emissions', 'soars', 'ages', 'george', 'ppm', 'same', 'torso', 'liter', 'samuel', 'duration', 'downgrades', 'savings', 'aveo', 'impellers', 'thickness', 'male', 'terahertz', 'flooding', 'higher', 'vinegar', 'fiscal', 'mccoy', 'again', 'pascals', 'loop', 'megabit', 'cregg', 'periodic', 'wpm', 'northeast', 'nh', 'people', 'teenie', 'children', 'lorazepam', 'sax', 'lufthansa', 'parallel', 'declination', 'salting', 'millimeters', 'current', 'temperature', 'soluble fiber content', 'mortality', 'francis', 'populations', 'hazardous waste generation', 'couple', 'geothermal', 'distance', 'fleurs', 'obesity rate', 'estimated', 'material footprint', 'erg', 'forecast', 'reference', 'NZD', 'gauss', 'seasonal', 'stack', 'far', 'mev', 'fat content', 'elevation', 'quarter', 'imrt', 'kilobits', 'starch content', 'volant', 'rain', 'femtosecond', 'moyne', 'lbp', 'farad', 'stronger', 'page', 'undesirably', 'ft', 'ubu', 'million', 'billions', 'oregano', 'minus', 'sustainable fisheries', 'aged', 'terabyte', 'centigrade', 'melting point', 'saturated fat content', 'microlitre', 'nth', 'northwest', 'enrollment rate', 'arity', 'computes', 'pontiac', 'units sold', 'ficus', 'stock', 'amd', 'oxidation stability', 'water scarcity', 'air quality index', 'arat', 'interest rate', 'mw', 'beck', 'kilowatts', 'diameter', 'mainly', 'premium', 'eoc', 'yards', 'gigahertz', 'life expectancy', 'jardin', 'cookie', 'megapascals', 'joblessness', 'iota', 'revenues', 'terabit', 'land degradation', 'density', 'expense', 'net', 'year', 'crime rate', 'p/e ratio', 'yield strength', 'oil content', 'groundwater depletion rate', 'rainfall', 'amk', 'water holding capacity', 'definition', 'compares', 'pecans', 'calculating', 'immigration rate', 'print', 'blaze', 'anemic', 'day', 'operating expenses', 'gallons', 'costs', 'beyond', 'robert', 'weiss', 'diameters', 'mbit', 'bits', 'kilograms', 'unemployement', 'biodiversity index', 'thick', 'generations', 'barely', 'decibels', 'motors', 'saturated', 'circumference', 'layers', 'boxes', 'weight', 'joseph', 'transparency of environmental reporting', 'libations', 'TWD', 'mopsus', 'pepper', 'cros', 'processor', 'weekend', 'milligrams', 'arthur', 'gbps', 'hertz', 'packages', 'leaving', 'chives', 'just', 'inr', 'acid deposition', 'bachelor', 'henry', 'kilometer', 'cups', 'terabits', 'milligrammes', 'pi', 'tau', 'omega', 'melting', 'cheese', 'hotter', 'combination', 'MYR', 'coming', 'mercury emissions', 'sitric', 'amount', 'ETH', 'kw', 'baud', 'square meters', 'nearly', 'habitat fragmentation', 'richard', 'clothing', 'kick', 'working capital', 'years', 'total debt', 'stormwater management', 'ghz', 'mudslides', 'rental rate', 'decline', 'herman', 'monsoon', 'charles', 'inventories', 'bernstein', 'tons', 'gwathmey', 'rather', 'rest', 'danta', 'glycemic index', 'fish stocks', 'strawberries', 'kbps', 'pound', 'veilleux', 'nanofarad', 'spewing', 'femminile', 'stimulus', 'vary', 'buyback', 'co2 emissions per capita', 'dried', 'nitrate content', 'pieces', 'prius', 'shorten', 'drank', 'kilocalories', 'becquerels', 'aeterna', 'population', 'milliseconds', 'school enrollment', 'downgraded', 'insurance', 'kgo', 'MXN', 'strength', 'southern', 'salinity level', 'sir', 'milliliters', 'doms', 'ftlbf', 'winds', 'daimler', 'millivolt', 'byte', 'pounds', 'approximate', 'efficiency', 'htc', 'longitude', 'cost', 'shareholders', 'deciliter', 'mold count', 'colder', 'gains', 'litres', 'mazur', 'pixel', 'reform', 'job satisfaction', 'calorie', 'encode', 'ranges', 'sulfur dioxide emissions', 'grain', 'seed count', 'probability', 'teaspoons', 'expectancy', 'joules', 'payout', 'turbocharged', 'reach', 'solid waste generation', 'invest', 'weller', 'accessories', 'width', 'fumbled', 'persons', 'eastern', 'unemployment rate', 'hour', 'sector', 'punt', 'phosphorus content', 'latitude', 'interest', 'shares', 'emr', 'off', 'diskette', 'snowfall', 'picocuries', 'guaranteed', 'sliced', 'processors', 'salary', 'when', 'budget', 'interception', 'ounce', 'flour protein content', 'data', 'threads', 'towns', 'surplus', 'millisieverts', 'nanoseconds', 'cleary', 'came', 'hence', 'decades', 'cores', 'nahai', 'unacceptably', 'dots', 'cooling', 'lasts', 'shorter', 'ounces per square inch', 'halflings', 'shortest', 'gram', 'almonds', 'trans fat content', 'insert', 'sandisk', 'qaisar', 'kmh', 'ohm', 'electricity consumption', 'pene', 'firms', 'beta', 'pickups', 'financial', 'urban sprawl', 'luminance', 'life', 'ratings', 'where', 'to', 'gundobad', 'theta', 'ochratoxin a level', 'energy efficiency', 'city', 'minute', 'free fatty acid content', 'addresses', 'INR', 'william', 'tax', 'payments', 'money', 'phi', 'weeks', 'megawatts', 'budgets', 'lactic acid bacteria count', 'infestation level', 'geographic', 'out', 'cooler', 'mwh', 'heavy metal pollution index', 'octane', 'btus', 'kib', 'centimeters', 'carbohydrate content', 'measured', 'electron volts', 'shallots', 'lending', 'eberhard', 'availability', 'dollars', 'daltons', 'iodine value', 'whey protein content', 'milliarcseconds', 'corresponds', 'electric current', 'hashish', 'annually', 'cocoa percentage', 'royce', 'silverado', 'radiofrequency', 'carbohydrates', 'microns', 'scheme', 'drought index', 'sales', 'kilometre', 'proofed', 'mu', 'defect count', 'vlahos', 'line', 'thyme', 'kilobit', 'maintain', 'tannin content', 'petrol', 'needs', 'weber', 'span', 'coldest', 'had', 'peak', 'chopped', 'ocean acidification index', 'next', 'teaspoon', 'bytes', 'shipped', 'bank', 'nne', 'droughts', 'generators', 'nutritional', 'anaemic', 'our', 'touchdowns', 'EUR', 'night', 'wetland health index', 'goods', 'feet', 'kilometeres', 'generates', 'paprika', 'boil', 'particulate matter concentration', 'engine', 'birth', 'vitamin a content', 'm', 'supercharged', 'liters', 'bpmn', 'megabyte', 'monochrome', 'forecasts', 'afternoon', 'vmax', 'foot', 'lundi', 'firestorms', 'tenskwatawa', 'mixture', 'payable', 'kev', 'printed', 'bennett', 'net income', 'precipitation', 'milepost', 'payment', 'kompany', 'increase', 'sprinkle', 'analysts', 'wreaked', 'toxic chemicals management', 'cilantro', 'invested', 'lists', 'extinguish', 'maintaining', 'cssd', 'megs', 'moisture content', 'banks', 'fluid ounces', 'bags', 'dots per inch', 'potassium content', 'investing', 'decreases', 'shafayat', 'budgetary', 'koss', 'land use change', 'lux', 'microplastic pollution', 'kms', 'mortgage', 'cannon', 'ecosystem services value', 'latitudes', 'consume', 'kilogrammes', 'depression score', 'BTC', 'john', 'kilogram', 'having', 'electricity consumption per capita', 'millihenry', 'celeron', 'skeat', 'winkleman', 'fig', 'soil erosion', 'equator', 'ph level', 'nutrient density', 'sublimation', 'fund', 'moisture', 'above', 'linguine', 'slowdown', 'sodium content', 'area', 'marine protected areas coverage', 'nvidia', 'om', 'carbs', 'product', 'total phenolic content', 'cloves', 'longitudes', 'rating', 'merchandise', 'newman', 'paying', 'volume', 'negative', 'plaintext', 'water pollution index', 'carob', 'sattva', 'beneath', 'karrer', 'alone', 'mannesmann', 'debt', 'megapixels', 'blood', 'shrinks', 'kirkeby', 'books', 'alarmingly', 'james', 'panasonic', 'celcius', 'karl', 'introduction', 'ecological footprint', 'caseload', 'preferably', 'employment rate', 'chevy', 'endocrine disruptors', 'engines', 'meters', 'gottlieb', 'air pollution', 'worth', 'rcc', 'wacc', 'ravaged', 'quality score', 'earnings', 'carapace', 'fiber length', 'arntzen', 'rainstorms', 'consumed', 'uses', 'serving size', 'sells', 'centrifuge', 'magazines', 'mbar', 'equivalent', 'landslides', 'located', 'shaw', 'preservative level', 'vitamin d content', 'kilos', 'carbon emissions', 'edward', 'gigaflops', 'diced', 'decade', 'JPY', 'manoir', 'cocaine', 'drams', 'adjacent', 'glut', 'shortfall', 'ionising', 'bonuses', 'daha', 'families', 'elizabeth', 'levels', 'degrees', 'tablespoon', 'methane emissions', 'chunks', 'powdered', 'sls', 'elongated', 'micrometers', 'megahertz', 'milliampere', 'selling', 'pesticide residue level', 'ball', 'cognitive ability', 'ulam', 'lbs', 'pension', 'need', 'ability', 'parsley', 'riou', 'heavy metal contamination', 'gas', 'particulate matter emissions', 'centimetre', 'dough', 'cm', 'werner', 'hefty', 'fabricio', 'eick', 'reduces', 'terms', 'specified', 'mustard', 'kegs', 'pints', 'balancing', 'upsilon', 'scoring', 'plummeting', 'gw', 'recycling rate', 'point', 'toricelli', 'affluence', 'hydropower', 'itt', 'laiho', 'roughly', 'mostly', 'micrograms', 'insoluble fiber content', 'kwh', 'river health index', 'grubs', 'USD', 'available', 'bulk', 'milligram', 'vitality score', 'fritz', 'stagnating', 'soil health index', 'twice', 'growth', 'gost', 'sizes', 'registers', 'decibel', 'predominantly', 'nuclear waste generation', 'waiting', 'genetic variation', 'than', 'easterlin', 'unpaid', 'wind speed', 'moody', 'macronutrient ratio', 'contract', 'one', 'beto', 'gasoline', 'athlon', 'mayonnaise', 'strips', 'range', 'separator', 'gain', 'buyout', 'taxpayers', 'autoregressive', 'level', 'depends', 'gigabytes', 'turmeric', 'rotates', 'enrollments', 'stockholders', 'heav', 'items', 'quart', 'modems', 'stock price', 'du', 'dimensions', 'inflation', 'lumens', 'water use efficiency', 'runs', 'urban heat island intensity', 'now', 'mbpd', 'comprise', 'food safety index', 'shipment', 'evening', 'serca', 'mpa', 'water quality index', 'annuity', 'suvs', 'stability', 'outlook', 'llamas', 'laptop', 'cornstarch', 'mwe', 'alpha', 'wages', 'per', 'putting', 'gwh', 'populated', 'priorities', 'hitting', 'minimum', 'cut', 'entries', 'sugar content', 'inflorescence', 'maximum', 'month', 'cranks', 'henchoz', 'excess', 'minced', 'CNY', 'besco', 'risk score', 'crape', 'shots', 'national parks coverage', 'curing time', 'kg', 'retailers', 'meiert', 'volts', 'kappa', 'nu', 'oxygen demand index', 'megabits', 'starting', 'storms', 'dietary fiber content', 'rains', 'GBP', 'fmu', 'litre', 'mb', 'bloch', 'newtons', 'favorability', 'papers', 'gigawatts', 'cubic feet', 'utility cost', 'flatus', 'mozzarella', 'celsius', 'degree', 'engel', 'northern', 'gallon', 'financing', 'flagon', 'centimeter', 'euros', 'increased', 'wheat gluten content', 'equation', 'opteron', 'gel strength', 'example', 'organic chemical pollution', 'hydroelectricity', 'schneider', 'thz', 'throughput', 'square millimeters', 'ev', 'speeding', 'map', 'census', 'least', 'volatile organic compound emissions', 'coar', 'mmx', 'older', 'total antioxidant capacity', 'ton', 'shortly', 'megajoules', 'persichetti', 'hourly wage', 'spoon', 'biochemical oxygen demand', 'variance', 'antibiotic use in livestock', 'fuel', 'cumin', 'sensory acceptability score', 'cent', 'previous', 'pounds per square inch', 'squash', 'gross profit', 'modem', 'friedrich', 'carbon dioxide concentration', 'touchdown', 'interceptions', 'meridian', 'eurytus', 'credit', 'item', 'resistor', 'natural disaster risk', 'caffeine', 'outputs', 'social security', 'taruna', 'flavius', 'nearby', 'gamma', 'legendre', 'on', 'microprocessors', 'rebate', 'megabytes', 'last', 'berdahl', 'dlrs', 'riccati', 'comparable', 'cubic', 'below', 'after', 'nanometres', 'low', 'caloric density', 'reducing', 'revolutions per minute', 'premiums', 'cazorla', 'capacity', 'lot size', 'candela', 'loans', 'nehrling', 'mpg', 'communities', 'durations', 'within', 'hasse', 'asset', 'equity', 'dry matter content', 'philip', 'climate change vulnerability', 'expectations', 'heating', 'book', 'measurements', 'morning', 'thawing time', 'saic', 'vlf']
    places = ['mono', 'sanilac', 'tishomingo', 'huntingdon', 'alexandria', 'newton', 'kandiyohi', 'coweta', 'cambodia', 'yakima', 'bahrain', 'wallowa', 'nauru', 'mason', 'yakutat', 'greensville', 'highland', 'kit', 'chicot', 'stokes', 'catoosa', 'chaffee', 'woodward', 'bolivar', 'brooke', 'cass', 'sherman', 'mccone', 'crowley', 'copiah', 'muscatine', 'dickson', 'powhatan', 'monaco', 'oceana', 'slope', 'sacramento', 'pasco', 'gregg', 'braxton', 'guam', 'cochran', 'edmonson', 'owyhee', 'barton', 'sequoyah', 'ketchikan', 'hertford', 'aguas', 'tuscaloosa', 'mongolia', 'milwaukee', 'ida', 'val', 'pickens', 'slovenia', 'cameron', 'transylvania', 'logan', 'ector', 'bulloch', 'botswana', 'ionia', 'bayfield', 'woods', 'sussex', 'upton', 'saline', 'walworth', 'ny', 'oceania', 'edgecombe', 'jennings', 'mahoning', 'vietnam', 'armstrong', 'dc', 'gulf', 'eastern', 'scioto', 'dorchester', 'juniata', 'st lucia', 'perry', 'pontotoc', 'trujillo', 'fayette', 'adjuntas', 'queens', 'wagoner', 'wyandotte', 'pottawatomie', 'chenango', 'swain', 'de', 'massachusetts', 'kimble', 'east timor', 'utuado', 'iraq', 'tuvalu', 'swains', 'eddy', 'kenedy', 'outagamie', 'aransas', 'grenada', 'floyd', 'emmons', 'netherlands', 'bradley', 'iosco', 'fall', 'bahamas', 'bulgaria', 'deer', 'massac', 'az', 'lehigh', 'payne', 'sc', 'greenwood', 'starr', 'mecklenburg', 'chippewa', 'russian federation', 'orleans', 'bingham', 'craig', 'central asia', 'tama', 'taos', 'russell', 'renville', 'volusia', 'dukes', 'mellette', 'llano', 'foard', 'arlington', 'wells', 'currituck', 'snyder', 'escambia', 'leavenworth', 'missaukee', 'fl', 'lowndes', 'leake', 'calloway', 'uintah', 'dickens', 'rensselaer', 'tuscola', 'abbeville', 'mcdowell', 'dolores', 'yazoo', 'bronx', 'brunei', 'indian', 'hennepin', 'luce', 'real', 'seward', 'east', 'lawrence', 'cayey', 'lares', 'cleburne', 'alamosa', 'nueces', 'falls', 'weakley', 'las', 'lycoming', 'alamance', 'dewey', 'juneau', 'hernando', 'stephens', 'washoe', 'wabash', 'guernsey', 'mccook', 'appling', 'mali', 'conejos', 'ware', 'charleston', 'teller', 'barron', 'monmouth', 'kanabec', 'blackford', 'piscataquis', 'roane', 'gilchrist', 'mills', 'bandera', 'tunica', 'yauco', 'mn', 'guadalupe', 'maunabo', 'sd', 'wake', 'shawano', 'perkins', 'day', 'deuel', 'bosnia herzegovina', 'pershing', 'atlantic', 'broomfield', 'bowman', 'allen', 'juana', 'northern america', 'clarion', 'gregory', 'macedonia', 'oh', 'wyoming', 'nc', 'holmes', 'beaverhead', 'nash', 'tioga', 'pipestone', 'graves', 'merced', 'rhode island', 'mayaguez', 'centre', 'patrick', 'miami', 'umatilla', 'bennington', 'deaf', 'avoyelles', 'rio', 'neshoba', 'sequatchie', 'mesa', 'cheboygan', 'motley', 'pawnee', 'nevada', 'vance', 'greece', 'kanawha', 'ga', 'stafford', 'kalkaska', 'presidio', 'roosevelt', 'latah', 'sioux', 'comanche', "o'brien", 'silver', 'barren', 'page', 'hillsborough', 'boundary', 'yankton', 'bayamon', 'chad', 'van', 'carlton', 'winneshiek', 'tunisia', 'turkey', 'kearney', 'musselshell', 'sully', 'jones', 'northern europe', 'labette', 'miner', 'harney', 'ziebach', 'nelson', 'norfolk', 'lancaster', 'switzerland', 'niger', 'pittsylvania', 'lesotho', 'glynn', 'sweet', 'seminole', 'nance', 'madagascar', 'ellsworth', 'vermillion', 'gage', 'meigs', 'tulsa', 'freestone', 'isabela', 'bangladesh', 'ohio', 'tyrrell', 'contra', 'united kingdom', 'de', 'pamlico', 'gaines', 'el', 'guyana', 'pickett', 'north america', 'seychelles', 'park', 'elk', 'maries', 'rosebud', 'haiti', 'north carolina', 'moultrie', 'morgan', 'idaho', 'edmunds', 'somervell', 'dewitt', 'ne', 'shannon', 'collingsworth', 'swaziland', 'hayes', 'butts', 'sandoval', 'garden', 'hendry', 'charles', 'whitfield', 'madera', 'hickman', 'jayuya', 'ascension', 'marshall', 'heard', 'worcester', 'gentry', 'person', 'mendocino', 'hunterdon', 'guatemala', 'ralls', 'nowata', 'portsmouth', 'kodiak', 'mountrail', 'independence', 'scotts', 'josephine', 'trimble', 'namibia', 'niagara', 'andorra', 'knott', 'nye', 'shackelford', 'kern', 'woodford', 'lander', 'loiza', 'lackawanna', 'columbus', 'coal', 'harper', 'bates', 'augusta', 'sutton', 'gilliam', 'luquillo', 'alcona', 'weld', 'wyandot', 'lumpkin', 'nodaway', 'cidra', 'philadelphia', 'san marino', 'coshocton', 'gordon', 'burke', 'western asia', 'dougherty', 'hyde', 'beltrami', 'moore', 'deschutes', 'jewell', 'estonia', 'gove', 'irion', 'toole', 'melanesia', 'robertson', 'mahnomen', 'brazoria', 'penobscot', 'turkmenistan', 'sheridan', 'sri lanka', 'geneva', 'williamsburg', 'maury', 'clinton', 'audrain', 'wakulla', 'adair', 'hampton', 'mower', 'cannon', 'latimer', 'treasure', 'hanover', 'cochise', 'itasca', 'houghton', 'oswego', 'arkansas', 'nicollet', 'albemarle', 'amelia', 'cullman', 'kiribati', 'chesterfield', 'sabana', 'granite', 'hamilton', 'shasta', 'chase', 'carter', 'haakon', 'woodruff', 'candler', 'greenlee', 'kershaw', 'northumberland', 'mahaska', 'hopewell', 'oregon', 'bennett', 'mclean', 'poinsett', 'muskegon', 'macon', 'wilkes', 'new jersey', 'pointe', 'grant', 'charlotte', 'beaver', 'bibb', 'androscoggin', 'maryland', 'izard', 'okmulgee', 'lexington', 'staunton', 'sonoma', 'mo', 'pondera', 'gonzales', 'montenegro', 'brown', 'pitt', 'stewart', 'missouri', 'king', 'germany', 'aroostook', 'pushmataha', 'coamo', 'ireland', 'posey', 'grayson', 'riley', 'minnehaha', 'lynchburg', 'columbiana', 'sarasota', 'swift', 'stutsman', 'carteret', 'dupage', 'parke', 'otter', 'humboldt', 'terrell', 'cabo', 'emery', 'qatar', 'burnet', 'valley', 'robeson', 'hale', 'san', 'republic', 'salem', 'cross', 'ks', 'tangipahoa', 'rhea', 'yell', 'rabun', 'johnson', 'broward', 'washakie', 'garvin', 'doniphan', 'spink', 'sierra', 'petroleum', 'unicoi', 'aibonito', 'walton', 'denali', 'shoshone', 'midland', 'moca', 'mauritania', 'iran', 'armenia', 'kankakee', 'pierce', 'goodhue', 'sabine', 'india', 'atoka', 'honduras', 'palm', 'benin', 'rota', 'polynesia', 'dundy', 'saluda', 'daviess', 'bon', 'caldwell', 'eastern europe', 'kent', 'culpeper', 'des', 'kay', 'korea north', 'pecos', 'archer', 'nassau', 'hansford', 'love', 'beckham', 'corson', 'haines', 'pinal', 'laramie', 'fannin', 'washita', 'allamakee', 'stone', 'harford', 'kansas', 'luna', 'cottle', 'denmark', 'dekalb', 'henderson', 'galveston', 'saipan', 'atascosa', 'martin', 'barnes', 'kazakhstan', 'sublette', 'central africa', 'broadwater', 'skagit', 'forrest', 'lemhi', 'grand', 'terry', 'vega', 'mecosta', 'swisher', 'neosho', 'berkeley', 'wilbarger', 'fleming', 'benewah', 'eagle', 'winkler', 'jo', 'co', 'treutlen', 'anne', 'rockcastle', 'lincoln', 'crittenden', 'crane', 'ukraine', 'chemung', 'tx', 'andrews', 'starke', 'howard', 'barranquitas', 'harding', 'drew', 'goochland', 'kossuth', 'isanti', 'amite', 'gasconade', 'toa', 'wright', 'lagrange', 'burkina', 'trinidad & tobago', 'hickory', 'laos', 'lucas', 'gillespie', 'united states', 'bureau', 'jenkins', 'indiana', 'gurabo', 'burleson', 'potter', 'costilla', 'reeves', 'salt', 'dawson', 'denton', 'carroll', 'zapata', 'id', 'washtenaw', 'arthur', 'liberia', 'dickenson', 'hockley', 'kaufman', 'dunn', 'cuyahoga', 'matanuska-susitna', 'bourbon', 'sitka', 'langlade', 'mccreary', 'lamb', 'schleicher', 'dale', 'giles', 'moldova', 'yadkin', 'skamania', 'live', 'bryan', 'multnomah', 'barnwell', 'alabama', 'catron', 'sanborn', 'mccracken', 'marlboro', 'hudson', 'dominica', 'lonoke', 'atchison', 'broome', 'grainger', 'hartley', 'siskiyou', 'clarke', 'itawamba', 'loudoun', 'hendricks', 'ri', 'northampton', 'south carolina', 'coryell', 'gilpin', 'breathitt', 'tompkins', 'millard', 'slovakia', 'estill', 'bond', 'indonesia', 'mckenzie', 'butler', 'lasalle', 'yukon-koyukuk', 'athens', 'erath', 'buena', 'alger', 'lubbock', 'nolan', 'chautauqua', 'scurry', 'sanpete', 'pope', 'cooper', 'vilas', 'calaveras', 'harrison', 'dillon', 'jay', 'gem', 'stonewall', 'york', 'bacon', 'anderson', 'zambia', 'lamoille', 'parmer', 'callahan', 'delta', 'bernalillo', 'iceland', 'nobles', 'tift', 'halifax', 'grays', 'pike', 'cottonwood', 'finland', 'darlington', 'hamlin', 'washburn', 'eau', 'graham', 'loving', 'albania', 'lenoir', 'oktibbeha', 'rutland', 'del', 'brantley', 'barceloneta', 'coffey', 'gladwin', 'dade', 'susquehanna', 'borden', 'brevard', 'ocean', 'adams', 'vieques', 'ventura', 'hudspeth', 'iron', 'eaton', 'mcdonald', 'vanuatu', 'mt', 'wisconsin', 'sudan', 'blanco', 'eritrea', 'garrard', 'new', 'riverside', 'mcnairy', 'traill', 'north', 'ottawa', 'fauquier', 'ben', 'guinea-bissau', 'trumbull', 'crenshaw', 'banner', 'maricao', 'watauga', 'snohomish', 'wolfe', 'alaska', 'angelina', 'johnston', 'gooding', 'citrus', 'beadle', 'goliad', 'waller', 'cheatham', 'luzerne', 'sao tome & principe', 'alpine', 'covington', 'livingston', 'lac', 'roberts', 'cabell', 'prairie', 'united arab emirates', 'otsego', 'pine', 'ok', 'villalba', 'duplin', 'hatillo', 'pueblo', 'pepin', 'titus', 'poquoson', 'cascade', 'fluvanna', 'cherokee', 'desha', 'arenac', 'wrangell', 'spokane', 'marion', 'lampasas', 'nottoway', 'kosovo', 'powder', 'pitkin', 'todd', 'wa', 'cook', 'door', 'guanica', 'columbia', 'hardin', 'utah', 'aurora', 'pendleton', 'cole', 'peru', 'dubois', 'juncos', 'belgium', 'dominican republic', 'barry', 'arapahoe', 'menard', 'sampson', 'nepal', 'worth', 'laurel', 'malawi', 'cooke', 'cherry', 'dinwiddie', 'calumet', 'story', 'codington', 'buchanan', 'cuba', 'ritchie', 'dare', 'navarro', 'schuyler', 'watonwan', 'prince', 'stoddard', 'carson', 'craven', 'vatican city', 'montgomery', 'guaynabo', 'ford', 'clatsop', 'licking', 'morris', 'bullock', 'marquette', 'eastern asia', 'dakota', 'manatee', 'bannock', 'gabon', 'pocahontas', 'fiji', 'reynolds', 'bell', 'fairfield', 'dixon', 'dorado', 'kleberg', 'towns', 'towner', 'colonial', 'cabarrus', 'gila', 'bee', 'kennebec', 'daniels', 'yabucoa', 'judith', 'marin', 'patillas', 'winchester', 'upson', 'mariposa', 'sargent', 'karnes', 'island', 'iredell', 'norman', 'naguabo', 'simpson', 'boyle', 'ashley', 'gray', 'walla', 'senegal', 'cyprus', 'frontier', 'foster', 'aguadilla', 'hempstead', 'concordia', 'comoros', 'fairbanks', 'burundi', 'frio', 'nm', 'petersburg', 'maui', 'edgar', 'shiawassee', 'caribou', 'hill', 'pima', 'georgia', 'stanly', 'fentress', 'rawlins', 'racine', 'mauritius', 'green', 'walker', 'wexford', 'dunklin', 'alfalfa', 'haskell', 'or', 'alcorn', 'yolo', 'attala', 'belknap', 'trigg', 'vanderburgh', 'runnels', 'routt', 'ia', 'kinney', 'allegan', 'hettinger', 'dauphin', 'genesee', 'jeff', 'kyrgyzstan', 'somerset', 'la', 'lafayette', 'vermilion', 'ma', 'baca', 'cheshire', 'williams', 'montour', 'barbour', 'obion', 'florida', 'dent', 'grundy', 'dawes', 'jamaica', 'young', 'linn', 'hampshire', 'cattaraugus', 'surry', 'scotland', 'rapides', 'northwest', 'box', 'chickasaw', 'jersey', 'kenton', 'southern africa', 'monterey', 'davie', 'archuleta', 'custer', 'berrien', 'blue', 'minnesota', 'summit', 'uzbekistan', 'keya', 'fresno', 'rose', 'creek', 'panola', 'esmeralda', 'richland', 'denver', 'new hampshire', 'thomas', 'taliaferro', 'michigan', 'st.', 'luxembourg', 'lafourche', 'charlevoix', 'caguas', 'mcmullen', 'wilkin', 'pa', 'hart', 'mississippi', 'meagher', 'quebradillas', 'allegany', 'sutter', 'pembina', 'crosby', 'hungary', 'strafford', 'weston', 'hunt', 'otoe', 'eastland', 'south africa', 'ontario', 'kittitas', 'navajo', 'harrisonburg', 'toombs', 'divide', 'martinsville', 'houston', 'somalia', 'wetzel', 'brookings', 'south-eastern asia', 'maldives', 'turner', 'micronesia', 'fallon', 'kosciusko', 'isle', 'ulster', 'buncombe', 'spotsylvania', 'egypt', 'habersham', 'meade', 'baker', 'okeechobee', 'tucker', 'osceola', 'calcasieu', 'trousdale', 'grafton', 'waukesha', 'concho', 'bedford', 'vt', 'bonner', 'pittsburg', 'elko', 'radford', 'audubon', 'billings', 'rwanda', 'hood', 'hooker', 'jasper', 'elkhart', 'wallace', 'pend', 'gosper', 'jackson', 'merrimack', 'mcclain', 'christian', 'storey', 'algeria', 'yemen', 'northern', 'brewster', 'corozal', 'clearwater', 'afghanistan', 'white', 'kalamazoo', 'wade', 'botetourt', 'coffee', 'baraga', 'jerauld', 'rains', 'tajikistan', 'colquitt', 'coconino', 'bollinger', 'rockland', 'schoharie', 'windsor', 'kerr', 'chouteau', 'crow', 'peoria', 'oxford', 'defiance', 'nd', 'sedgwick', 'bertie', 'kewaunee', 'queen', 'northern asia', 'spain', 'warrick', 'larimer', 'morrison', 'lake', 'ravalli', 'myanmar', 'calhoun', 'vermont', 'alexander', 'kane', 'canovanas', 'belarus', 'curry', 'briscoe', 'yamhill', 'jefferson', 'ponce', 'mozambique', 'wicomico', 'lanier', 'bland', 'sweden', 'saguache', 'elmore', 'henrico', 'waldo', 'cayuga', 'philippines', 'tehama', 'milam', 'dane', 'gates', 'howell', 'klamath', 'freeborn', 'rappahannock', 'kenai', 'putnam', 'pakistan', 'saratoga', 'oconee', 'harris', 'clackamas', 'isabella', 'saint vincent & the grenadines', 'madison', 'chisago', 'loudon', 'meriwether', 'wythe', 'sumner', 'parker', 'allendale', 'ashe', 'colombia', 'yuba', 'manassas', 'lyman', 'miami-dade', 'lapeer', 'midway', 'whitman', 'fillmore', 'burnett', 'clearfield', 'wheeler', 'hamblen', 'cumberland', 'oakland', 'winnebago', 'okaloosa', 'bath', 'thailand', 'clare', 'buckingham', 'benson', 'mccurtain', 'richmond', 'bartholomew', 'evangeline', 'south america', 'pearl', 'lithuania', 'wheatland', 'clermont', 'caroline', 'camuy', 'southern europe', 'oscoda', 'hot', 'wabaunsee', 'macomb', 'roanoke', 'south sudan', 'tuscarawas', 'little', 'monona', 'wichita', 'matagorda', 'yoakum', 'shenandoah', 'glacier', 'davidson', 'gambia', 'ethiopia', 'west virginia', 'metcalfe', 'price', 'avery', 'fredericksburg', 'horry', 'guilford', 'travis', 'liberty', 'randolph', 'bear', 'plumas', 'equatorial guinea', 'mayes', 'prentiss', 'cassia', 'wayne', 'mille', 'rutherford', 'wirt', 'eureka', 'yellowstone', 'czech republic', 'manitowoc', 'fulton', 'camp', 'natrona', 'galax', 'will', 'saudi arabia', 'muskogee', 'twiggs', 'lipscomb', 'thurston', 'fort', 'alpena', 'wahkiakum', 'george', 'wi', 'cowley', 'bristol', 'hinsdale', 'gwinnett', 'hardee', 'troup', 'carbon', 'winona', 'tipton', 'winston', 'palo', 'dallas', 'caddo', 'pinellas', 'wharton', 'bracken', 'morrow', 'europe', 'bleckley', 'sheboygan', 'mexico', 'natchitoches', 'holt', 'meeker', 'hartford', 'rice', 'medina', 'torrance', 'antigua & deps', 'flathead', 'cleveland', 'telfair', 'lebanon', 'mcduffie', 'smyth', 'scott', 'minidoka', 'traverse', 'newaygo', 'argentina', 'arecibo', 'oldham', 'taney', 'wibaux', 'atkinson', 'hi', 'bastrop', 'becker', 'keokuk', 'willacy', 'gallatin', 'maine', 'passaic', 'wasco', 'schley', 'ut', 'wilson', 'rockwall', 'kings', 'mifflin', 'humphreys', 'chambers', 'edwards', 'marengo', 'ms', 'chilton', 'socorro', 'onondaga', 'tanzania', 'cheyenne', 'oliver', 'ivory coast', 'tattnall', 'antelope', 'stanley', 'jack', 'menifee', 'muhlenberg', 'sagadahoc', 'golden', 'samoa', 'davison', 'rockbridge', 'tulare', 'vigo', 'larue', 'glenn', 'summers', 'bhutan', 'whiteside', 'wv', 'djibouti', 'owsley', 'crockett', 'dickey', 'south dakota', 'cerro', 'ballard', 'sierra leone', 'nigeria', 'craighead', 'carlisle', 'la', 'bay', 'mcpherson', 'catahoula', 'barbados', 'guthrie', 'salinas', 'placer', 'inyo', 'jessamine', 'coles', 'valencia', "manu'a", 'sumter', 'colfax', 'clinch', 'vinton', 'hancock', 'chaves', 'colorado', 'wilcox', 'california', 'echols', 'preston', 'nicholas', 'cedar', 'anasco', 'harlan', 'bremer', 'coahoma', 'manistee', 'roscommon', 'leflore', 'brule', 'castro', 'rockingham', 'duchesne', 'victoria', 'bucks', 'newberry', 'talbot', 'anoka', 'oglethorpe', 'whatcom', 'searcy', 'santa', 'norway', 'beaufort', 'accomack', 'nicaragua', 'lewis', 'lunenburg', 'casey', 'mi', 'mobile', 'stanton', 'levy', 'tripp', 'decatur', 'solano', 'trempealeau', 'trinity', 'aiken', 'brazos', 'geauga', 'limestone', 'grimes', 'ouachita', 'ciales', 'new york', 'brunswick', 'prowers', 'tippecanoe', 'phillips', 'nacogdoches', 'bladen', 'herkimer', 'uinta', 'screven', 'sac', 'knox', 'amador', 'lea', 'fond', 'nez', 'otero', 'al', 'cuming', 'ramsey', 'gilmer', 'power', 'ashtabula', 'colbert', 'bolivia', 'bent', 'montmorency', 'benton', 'cortland', 'virginia', 'clay', 'middlesex', 'mclennan', 'st kitts & nevis', 'lavaca', 'elliott', 'andrew', 'moody', 'baylor', 'ozark', 'cape verde', 'bullitt', 'suwannee', 'lamar', 'blount', 'henry', 'iroquois', 'portugal', 'mccormick', 'missoula', 'koochiching', 'issaquena', 'rusk', 'twin', 'sweetwater', 'major', 'ar', 'essex', 'arroyo', 'douglas', 'chowan', 'nh', 'miller', 'arizona', 'japan', 'dutchess', 'garza', 'spencer', 'cecil', 'menominee', 'greer', 'mcleod', 'glascock', 'lamoure', 'sevier', 'venango', 'cibola', 'greenbrier', 'murray', 'schoolcraft', 'colusa', 'schuylkill', 'big', 'quay', 'quitman', 'edgefield', 'terrebonne', 'randall', 'amherst', 'malaysia', 'poweshiek', 'morrill', 'steuben', 'burlington', 'huntington', 'kauai', 'wasatch', 'kingfisher', 'harnett', 'windham', 'kimball', 'venezuela', 'cobb', 'israel', 'ontonagon', 'malheur', 'highlands', 'hawkins', 'desoto', 'orocovis', 'le', 'ogle', 'wise', 'kiowa', 'romania', 'catawba', 'pennsylvania', 'wadena', 'conecuh', 'roger', 'guayanilla', 'morovis', 'benzie', 'huerfano', 'huron', 'bowie', 'upshur', 'faulk', 'grady', 'ecuador', 'glades', 'davis', 'southern asia', 'fergus', 'sherburne', 'tinian', 'kendall', 'seneca', 'new mexico', 'chattooga', 'etowah', 'bradford', 'letcher', 'alachua', 'faulkner', 'hanson', 'reno', 'macoupin', 'pottawattamie', 'stearns', 'sterling', 'mohave', 'hillsdale', 'western europe', 'uruguay', 'canada', 'hubbard', 'coosa', 'ct', 'azerbaijan', 'pulaski', 'kootenai', 'magoffin', 'greene', 'austin', 'uvalde', 'noble', 'long', 'gogebic', 'black', 'schenectady', 'tippah', 'canadian', 'marinette', 'rich', 'phelps', 'france', 'cloud', 'ste.', 'paulding', 'gallia', 'boulder', 'laporte', 'weber', 'bartow', 'wood', 'barrow', 'oklahoma', 'ada', 'norton', 'glasscock', 'bexar', 'north dakota', 'kitsap', 'charlton', 'suffolk', 'crook', 'saunders', 'gibson', 'sibley', 'jerome', 'sangamon', 'pettis', 'berkshire', 'morton', 'wabasha', 'zavala', 'bamberg', 'plaquemines', 'yuma', 'harmon', 'belize', 'caswell', 'butte', 'montana', 'greenville', 'monongalia', 'solomon islands', 'connecticut', 'boyd', 'wapello', 'franklin', 'lyon', 'lorain', 'wy', 'dearborn', 'lauderdale', 'colleton', 'spalding', 'fremont', 'coleman', 'eastern africa', 'hitchcock', 'durham', 'moffat', 'presque', 'taylor', 'leon', 'appomattox', 'sharp', 'honolulu', 'taiwan', 'camas', 'chattahoochee', 'southampton', 'james', 'kenya', 'md', 'pasquotank', 'singapore', 'raleigh', 'faribault', 'charlottesville', 'olmsted', 'laclede', 'ca', 'iberville', 'serbia', 'hemphill', 'orangeburg', 'bledsoe', 'pratt', 'sullivan', 'yancey', 'acadia', 'westmoreland', 'mcintosh', 'stanislaus', 'burleigh', 'hawaii', 'el salvador', 'mckean', 'yavapai', 'tennessee', 'kuwait', 'hall', 'ouray', 'dallam', 'hopkins', 'piute', 'tate', 'western africa', 'canyon', 'morocco', 'va', 'steele', 'district', 'walsh', 'spartanburg', 'garfield', 'humacao', 'me', 'laurens', 'rooks', 'ness', 'marshall islands', 'penuelas', 'converse', 'kidder', 'oconto', 'hughes', 'frederick', 'valdez-cordova', 'kalawao', 'tensas', 'texas', 'africa', 'erie', 'piatt', 'western', 'choctaw', 'ferry', 'bosque', 'kemper', 'montrose', 'peach', 'antrim', 'tyler', 'barnstable', 'monroe', 'providence', 'modoc', 'albany', 'angola', 'lee', 'ingham', 'richardson', 'duval', 'tonga', 'yellow', 'doddridge', 'rincon', 'west', 'williamson', 'costa rica', 'tolland', 'clayton', 'lenawee', 'waupaca', 'hidalgo', 'stephenson', 'maverick', 'webb', 'daggett', 'comerio', 'nuckolls', 'tillman', 'skagway', 'gaston', 'tom', 'marathon', 'cambria', 'belmont', 'garland', 'bonneville', 'flagler', 'hodgeman', 'sarpy', 'china', 'fairfax', 'sauk', 'auglaize', 'berks', 'sandusky', 'boise', 'stillwater', 'culebra', 'evans', 'hocking', 'overton', 'camden', 'powell', 'newport', 'finney', 'haywood', 'mchenry', 'garrett', 'lynn', 'cotton', 'rock', 'allegheny', 'manati', 'los', 'lassen', 'apache', 'dixie', 'lane', 'thayer', 'sanders', 'rockdale', 'keweenaw', 'mathews', 'morehouse', 'campbell', 'greeley', 'trego', 'napa', 'vernon', 'liechtenstein', 'calvert', 'anchorage', 'early', 'austria', 'bienville', 'dyer', 'collier', 'ceiba', 'platte', 'mingo', 'pacific', 'forest', 'ward', 'harvey', 'clark', 'montezuma', 'louisa', 'tn', 'blaine', 'winn', 'hoke', 'branch', 'niobrara', 'waynesboro', 'donley', 'muscogee', 'coke', 'autauga', 'oman', 'champaign', 'redwood', 'burma', 'loup', 'aguada', 'cache', 'chile', 'kingman', 'georgetown', 'pemiscot', 'dooly', 'osage', 'ray', 'nj', 'callaway', 'uganda', 'bethel', 'comal', 'chatham', 'dona', 'alameda', 'burt', 'panama', 'elbert', 'shawnee', 'smith', 'rankin', 'ak', 'wilkinson', 'mcculloch', 'saginaw', 'ashland', 'boone', 'baltimore', 'effingham', 'churchill', 'yates', 'gunnison', 'mackinac', 'korea south', 'union', 'ross', 'ogemaw', 'waushara', 'merrick', 'addison', 'nebraska', 'haralson', 'okfuskee', 'baxter', 'brooks', 'chariton', 'cavalier', 'hays', 'crawford', 'clear', 'crisp', 'montcalm', 'gadsden', 'rush', 'orange', 'kingsbury', 'dubuque', 'griggs', 'dimmit', 'irwin', 'tuolumne', 'hinds', 'hutchinson', 'il', 'pender', 'hoonah-angoon', 'litchfield', 'cowlitz', 'banks', 'mineral', 'chittenden', 'darke', 'bailey', 'cimarron', 'washington', 'yalobusha', 'caribbean', 'syria', 'childress', 'italy', 'kearny', 'tazewell', 'maricopa', 'moniteau', 'dickinson', 'mercer', 'florence', 'asia', 'westchester', 'northern africa', 'klickitat', 'chelan', 'culberson', 'tillamook', 'warren', 'mcminn', 'tarrant', 'greenup', 'ringgold', 'kittson', 'dodge', 'alleghany', 'tooele', 'nv', 'cape', 'stark', 'nemaha', 'tallapoosa', 'furnas', 'baldwin', 'australia', 'barber', 'granville', 'malta', 'mora', 'reagan', 'sebastian', 'bergen', 'illinois', 'iowa', 'ellis', 'oneida', 'imperial', 'beauregard', 'hand', 'kenosha', 'congo', 'ochiltree', 'tallahatchie', 'okanogan', 'conway', 'owen', 'central african rep', 'fajardo', 'southeast', 'emporia', 'jordan', 'hormigueros', 'appanoose', 'gratiot', 'rogers', 'leslie', 'keith', 'noxubee', 'sharkey', 'clallam', 'leelanau', 'montague', 'anson', 'aleutians', 'coos', 'nantucket', 'plymouth', 'poland', 'shelby', 'ghana', 'pennington', 'louisiana', 'libya', 'juab', 'walthall', 'kentucky', 'palau', 'polk', 'chester', 'mitchell', 'red', 'togo', 'nome', 'ripley', 'forsyth', 'rowan', 'pickaway', 'danville', 'cameroon', 'waseca', 'australia and new zealand', 'sunflower', 'hampden', 'bossier', 'buffalo', 'payette', 'dillingham', 'jim', 'guayama', 'ransom', 'emanuel', 'aitkin', 'preble', 'claiborne', 'throckmorton', 'guinea', 'district of columbia', 'whitley', 'fountain', 'refugio', 'ozaukee', 'mckinley', 'clarendon', 'collin', 'papua new guinea', 'blair', 'carver', 'porter', 'assumption', 'teton', 'croatia', 'hardeman', 'sawyer', 'rolette', 'webster', 'iberia', 'perquimans', 'geary', 'fisher', 'roseau', 'in', 'mcdonough', 'cocke', 'suriname', 'catano', 'gloucester', 'chesapeake', 'lajas', 'breckinridge', 'woodson', 'zimbabwe', 'paraguay', 'stevens', 'hardy', 'caledonia', 'naranjito', 'onslow', 'woodbury', 'delaware', 'talladega', 'emmet', 'portage', 'new zealand', 'osborne', 'latvia', 'asotin', 'muskingum', 'bottineau', 'pleasants', 'central america', 'carolina', 'ky', 'goshen']

    # ----- Load pre=computed groups -----
    with open(config.categoricals_file, "r") as f:
        similar_word_map = json.load(f)
        similar_word_map = {int(k): v for k, v in similar_word_map.items()}

    # Possible parameter combinations
    remove_spines = [True, False]
    remove_ticks = [True, False]
    bar_colors = ["lightcoral", "firebrick", "coral", 
                "peru", "yellowgreen", "green",
                "forestgreen", "steelblue", "dodgerblue", 
                "skyblue", "lightsteelblue", "plum", 
                "darkblue", "mediumvioletred", "lightpink",
                'tab:blue','tab:orange','tab:green',
                'tab:red','tab:purple','tab:brown',
                'tab:pink','tab:gray','tab:olive','tab:cyan',
                ]
    mpl_styles = ["_mpl-gallery", "_mpl-gallery-nogrid", "dark_background", "fivethirtyeight", "ggplot", "seaborn-v0_8-whitegrid"]
    font_families = ["Microsoft Sans Serif", "Calibri", "Arial", "Times New Roman", "Comic Sans MS"]
    font_families = ["Calibri", "DejaVu Sans", "Tahoma", "Verdana"]
    font_sizes = [8, 10, 12, 14]
    edge_colors = ["none", "black"]
    font_styles = ["normal", "italic"]
    font_weights = ["normal", "bold"]
    series_sizes = range(4, 12)


    # Seed for reproducability
    np.random.seed(config.seed)

    # Create IMGs
    create_vertical_bars(config.generate_n_imgs)