import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Industrial_time_series_analysis.Describe.describe_utils.mocar_util.path_methods.DistanceCalculatorMethod import DistanceCalculatorMethod

class Methods:

    def __init__(self):

        from Industrial_time_series_analysis.Describe.describe_utils.mocar_util.path_methods.DijkstraMethod import DijkstraMethod
        from Industrial_time_series_analysis.Describe.describe_utils.mocar_util.path_methods.FloydWarshallMethod import FloydWarshallMethod
        from Industrial_time_series_analysis.Describe.describe_utils.mocar_util.path_methods.LandmarksMethod import LandmarksMethod

        self.methods = {
            'L': LandmarksMethod,
            'FW': FloydWarshallMethod,
            'D': DijkstraMethod
        }

    def byName(self, name, fermat) -> DistanceCalculatorMethod:
        if name in self.methods.keys():
            return self.methods[name](fermat)
        else:
            raise Exception('Unknown method name: {}'.format(name))
