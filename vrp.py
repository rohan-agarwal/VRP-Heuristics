from functions import *


def main():
    # Initialization
    loc = csv_to_np(location_file)
    demand = csv_to_np(demand_file)
    dist = csv_to_np(distance_file)
    for i in range(len(dist)):
        dist[i, i] = 99

    frequency = dtof(demand)
    day1, day2, f1, f2 = day_split(loc, frequency)

    # Kmeans
    labels1, labels2 = kmeans_wrapper(day1, day2, 4)

    # Split into clusters and solve TSP
    dc1, dc2, df1, df2 = split_wrapper(loc, day1, labels1, day2, labels2, dist)
    times, paths, total_time = tsp_shuffle_wrapper(
        dc1, dc2, df1, df2, labels1, labels2, day1, day2, dist, loc)

    # Update step: Can we increase visit frequency and reduce total time?
    new_frequency = list(frequency)
    print frequency
    for i in range(len(new_frequency)):
        new_frequency = list(frequency)
        if new_frequency[i] == 2:
            new_frequency[i] = 3
            new_day1, new_day2, new_f1, new_f2 = day_split(loc, new_frequency)
            new_l1, new_l2 = kmeans_wrapper(new_day1, new_day2, 4)
            ndc1, ndc2, ndf1, ndf2 = split_wrapper(
                loc, new_day1, new_l1, new_day2, new_l2, dist)
            ntimes, npaths, n_total_time = tsp_shuffle_wrapper(
                ndc1, ndc2, ndf1, ndf2, new_l1, new_l2, new_day1, new_day2, dist, loc)
            if n_total_time < total_time:
                total_time = n_total_time
                frequency = new_frequency
                paths = npaths
                times = ntimes

    # Final outputs
    print "Frequencies: "
    print frequency
    print "Paths: "
    for i in paths:
        print i
    print "Times: "
    print times
    print "Total time: " + str(total_time)

    return frequency, paths, times, total_time

main()
