spritepath = 'sprites/'

objects = {
    'grass': {
        'index': 0,
        'value': 0,
        'sprite': 'sprites/grass.png',
        'background': True,
        },
    'puddle': {
        'index': 1,
        'value': -1,
        'sprite': 'sprites/puddle.png',
        'background': True,
        },
    'star': {
        'index': 2, 
        'value': 0,
        'sprite': 'sprites/star.png',
        'background': False,
        },
    'circle': {
        'index': 3, 
        'value': 0,
        'sprite': 'sprites/circle.png',
        'background': False,
        },
    'square': {
        'index': 4,
        'value': 0,
        'sprite': 'sprites/square.png',
        'background': False,
        },
    'octagon': {
        'index': 5,
        'value': 0,
        'sprite': 'sprites/octagon.png',
        'background': False,
        },
    'trapezoid': {
        'index': 6,
        'value': 0,
        'sprite': 'sprites/trapezoid.png',
        'background': False,
        }
}

directions = {
    'to top left of': (-1, -1),
    'on top of': (-1, 0),
    'to top right of': (-1, 1),
    'to left of': (0, -1),
    'with': (0, 0),
    'to right of': (0, 1),
    'to bottom left of': (1, -1),
    'on bottom of': (1, 0),
    'to bottom right of': (1, 1)
}

background = 'sprites/grass.png'

# print objects