int add_int(int *start, int *len){
    int result = 0;
    for (int var = 0; var < *len; ++var) {
        result += start[var];
    }
    return result;
}
