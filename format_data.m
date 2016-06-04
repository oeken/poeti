clc;
clear;
close all;

filename = 'dataset.txt';
A = importdata(filename);
A = reshape(A,[2,192]);
A = A';
csvwrite('dataset.csv',A)
% cell2csv('nba.csv', whole)
