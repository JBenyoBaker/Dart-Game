class dart:
    locations = [] 
    x_values = []
    y_values = []
    def __init__(self, x_values, y_values):
        self.locations = []
        self.x_values = x_values
        self.y_values = y_values


    def set_points(self):
        for i in range(4):
            self.locations.append([self.x_values[0], self.y_values[0]])
            self.x_values = self.x_values[1:]
            self.y_values = self.y_values[1:]
    
    def move_points(self):
        #print(self.x_values)
        #print(self.y_values)
        self.locations.append([self.x_values[0], self.y_values[0]])
        self.x_values = self.x_values[1:]
        self.y_values = self.y_values[1:]
        self.locations = self.locations[1:]

    def get_x(self):
        return self.locations[0][0]
    
    def get_locations(self):
        return self.locations