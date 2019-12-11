jab1 = load('/nrs/branson/kwaki/jab_exp/M134w_20150427.jab', '-mat');
jab1 = jab1.x;
% jab2 = load('/nrs/branson/kwaki/jab_exp/M134w_20150505.jab', '-mat');
jab2 = load('/nrs/branson/kwaki/jab_exp/M174_20150416.jab', '-mat');
jab2 = jab2.x;


[diffs, a, b] = jab_diff(jab1, jab2)
