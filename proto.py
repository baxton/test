

import numpy as np
import scipy as sp
from sklearn.linear_model import SGDClassifier








class QuakePredictor(object):

    def __init__(self):
        self.cl = SGDClassifier("log")
	self.hour_data = []



    def is_quake(self, site_loc, quakes):
        count = 0.

        MAX_DIST = 70
  
        N = len(quakes)

        for i in range(N/5):
            lat = quakes[i * 5]
            lon = quakes[i * 5 + 1]
            mag = quakes[i * 5 + 3]

            dist = self.distance(site_loc, (lat, lon))

            if dist <= MAX_DIST:
                count += 1.

        return 0. if count < 2 else 1.
             



    def distance(self, loc1, loc2):
        """
        location = (latitude, longitude)
        """
        earthRadius = 6371.01
        
        deltaLon = np.abs( loc1[1] - loc2[1] )
        if deltaLon > 180:
       	    deltaLon = 360 - deltaLon

        dist = earthRadius * np.arctan2( np.sqrt( ( np.cos(loc1[0]) * np.sin(deltaLon) )**2 + ( np.cos(loc2[0]) * np.sin(loc1[0]) - np.sin(loc2[0]) * np.cos(loc1[0]) * np.cos(deltaLon) )**2 ), np.sin(loc2[0]) * np.sin(loc1[0]) + np.cos(loc2[0]) * np.cos(loc1[0]) * np.cos(deltaLon) )

        return dist





    def init(self, sampleRate, numOfSites, sitesData):
        self.rate = sampleRate
        self.sites = numOfSites
        self.sitesLocations = sitesData

        for c in range(self.sites):
            self.hour_data.append([])


    def feed_data_to_classifier(self):
        FEATURES = 0
        EVENTS = 1

        HS = len(self.hour_data[0])  # number of hous passed

        # features forms:
        # [0]    - time in the future, zero based
        # [1..N] - features

        for s in range(self.sites):
            for h in range(HS):
                # predict for the current hour
                X = np.concatenate(([0], self.hour_data[s][h][FEATURES]))
                Y = self.hour_data[s][h][EVENTS]
                self.cl.partial_fit(X, Y)

                # for future predictions
                for hf in range(h, HS):
                    X[0] = hf
                    Y = self.hour_data[s][hf][EVENTS]
                    self.cl.partial_fit(X, Y)
    


   
    def get_features(self, data, begin):
        N = self.rate * 3600
        d = data[begin : begin + N * s].reshape((3, N))

        m = np.mean(d, axis=1)
        v = np.var(d, axis=1)
        k = sp.stats.kurtosis(d, axis=1)
        s = sp.stats.skew(d, axis=1)

        features = sp.concatenate((m.reshape((3,1)), v.reshape((3,1)), k.reshape((3,1)), s.reshape((3,1))), axis=1)

        return features.reshape((12,))
	


    def forecast(self, hour, data, K, globalQuakes):
        return_size = self.sites * 2160
        predictions = [0.] * return_size

        for s in range(self.sites):
            beg = s * self.rate * 3600 * 3

            features = self.get_features(data, beg) 
            events = np.array([self.is_quake((self.siteLocations[s*2], self.siteLocations[s*2+1]), globalQuakes)])

            self.hour_data[s].append( (features, events) )

            self.feed_data_to_classifier()

            # predict
            for h in range(hour, 2160):
                t = h - hour
                X = np.concatenate(([t], features))
                p = self.cl.predict(X)
                predictions[t * self.sites + s] = p
        return predictions






def main():
    o = QuakePredictor()

    # distance from Moscow to Singapore about 6K km
    loc1 = (37.6156, 55.7522)
    loc2 = (103.8, 1.3667)
    d = o.distance(loc1, loc2)
    print d


#main()
