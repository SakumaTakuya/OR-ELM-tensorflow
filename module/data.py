class TimeSeriesData:
    def __init__(self, data, interval):
        self.index = 0
        def next_data(data):
            for d in data:
                self.index += 1
                yield d

        self.interval = interval
        self.next_data = next_data(data)

        self.current_time = 0
        self.current_data = next(self.next_data)

    def get_next_data(self, time, default=None):
        """
            return data at {time} seconds.
        """
        prev_time = self.current_time
        prev_data = self.current_data

        dat = next(self.next_data, default)

        if dat is default:
            return default

        self.current_time += self.interval
        self.current_data = dat

        if self.current_time >= time:
            section = self.current_time - prev_time
            diff = self.current_time - time
            t = diff / section
            return t * prev_data + (1 - t) * self.current_data
        else:
            return self.get_next_data(time)    