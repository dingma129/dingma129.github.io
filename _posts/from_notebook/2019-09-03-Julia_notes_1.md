---
title: "Julia: Notes-1"
layout: splash
excerpt: "Basics"
categories: [Julia]
---

# 1. Data Structures

### tuple


```julia
a = ("a","b","c")
# 1-based indexing
@show a[1];
# immutable
```

    a[1] = "a"


### dictionary


```julia
b = Dict("one"=>1,"two"=>2,"three"=>3)
@show b["two"]
# adding another item
b["four"] = 4
@show b
# can also use push
@show push!(b,"five"=>5);
```

    b["two"] = 2
    b = Dict("two" => 2,"four" => 4,"one" => 1,"three" => 3)
    push!(b, "five" => 5) = Dict("two" => 2,"four" => 4,"one" => 1,"three" => 3,"five" => 5)


### array


```julia
c = ["red","blue","yellow"]
@show c
# push: append
@show push!(c,"green")  # push and push! returns a new array
@show c
@show pop!(c)  # pop and pop! returns the popped element
@show c;
```

    c = ["red", "blue", "yellow"]
    push!(c, "green") = ["red", "blue", "yellow", "green"]
    c = ["red", "blue", "yellow", "green"]
    pop!(c) = "green"
    c = ["red", "blue", "yellow"]


### n-dim array


```julia
@show d1 = [[1,2,3],[4,5,6],[7,8,9]]
@show typeof(d1)  # Array{Array{Int64,1},1}

@show d2 = [1 2 3;4 5 6;7 8 9] 
@show typeof(d2)  # Array{Int64,2}

# convert from Array{Array{Int64,1},1} to Array{Int64,2}
@show Array{Int64,2}(hcat(d1...)')

# array assignment is by reference
d3 = d1
d3[1][1] = -1
@show d1  # (1,1)-th element 1 => -1

# for array, copy() is enough
d4 = copy(d1)
d4[1] = [-1,-2,3]
@show d1  # using copy: 1st element [-1,2,3] does not change into [-1,-2,3]

# for array of arrays, copy() only copies the first level
d5 = copy(d1)
d5[1][2] = -2
@show d1  # using copy: (1,2)-th element 2 => -2

# need to use deepcopy()
d6 = deepcopy(d1)
d6[1][3] = -3
@show d1;  # using deepcopy: (1,3)-th element 3 does not change into -3
```

    d1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]] = Array{Int64,1}[[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    typeof(d1) = Array{Array{Int64,1},1}
    d2 = [1 2 3; 4 5 6; 7 8 9] = [1 2 3; 4 5 6; 7 8 9]
    typeof(d2) = Array{Int64,2}
    Array{Int64, 2}((hcat(d1...))') = [1 2 3; 4 5 6; 7 8 9]
    d1 = Array{Int64,1}[[-1, 2, 3], [4, 5, 6], [7, 8, 9]]
    d1 = Array{Int64,1}[[-1, 2, 3], [4, 5, 6], [7, 8, 9]]
    d1 = Array{Int64,1}[[-1, -2, 3], [4, 5, 6], [7, 8, 9]]
    d1 = Array{Int64,1}[[-1, -2, 3], [4, 5, 6], [7, 8, 9]]


# 2. Control Flow

### for loop


```julia
# use =
for i = 1:3
    print("$i ")
end
println()
# use in
for i in 4:6
    print("$i ")
end
println()
```

    1 2 3 
    4 5 6 



```julia
# multiple for loops
for i in 3:4, j in 5:6
    println("$i * $j = $(i*j)")
end
```

    3 * 5 = 15
    3 * 6 = 18
    4 * 5 = 20
    4 * 6 = 24


### if-elseif-else


```julia
for e in [1,3,5]
    if e < 3
        println("$e is less than 3")
        elseif e > 3
        println("$e greater than 3")
    else
        println("$e exactly 3")
    end
end
```

    1 is less than 3
    3 exactly 3
    5 greater than 3


# 3. Function

### define a function


```julia
f1(x,y) = x+y
@show f1(1,1)

# anonymous function
f2 = (x,y) -> x+y
@show f2(1,1)

# if there is no "return", it returns the value in the last line
function f3(x,y)
    x+y
end
@show f3(1,1)
```

    f1(1, 1) = 2
    f2(1, 1) = 2
    f3(1, 1) = 2





    2



### multiple dispatch
A key feature of Julia is multiple dispatch. 

Julia will dispatch onto the strictest acceptible type signature.


```julia
# 3 different argument types 
function f(x::Int64)
    println("$x is Int")
    return x+10
end

function f(x::Float64)
    println("$x is Float")
    return x-10
end

function f(x::String)
    println("$x is String")
    return x*"10"
end

@show f(0)
@show f(0.0)
@show f("0")

# f (generic function with 3 methods)
f
```

    0 is Int
    f(0) = 10
    0.0 is Float
    f(0.0) = -10.0
    0 is String
    f("0") = "010"





    f (generic function with 3 methods)



### function with multiple return types


```julia
# function can return multiple types
function g(x,y;z)  # z is an optional argument
    if z > 0
        return (x,y,x+y)
    else
        return [x,y]
    end
end

@show g(1,2;z=3)    # returns tuple
@show g(1,2;z=-3);  # returns array
```

    g(1, 2; z=3) = (1, 2, 3)
    g(1, 2; z=-3) = [1, 2]


### function in function


```julia
# a basic example of tail recursive version of fibonacci number
function fib(n::Int64)::BigInt
    # tail-rec
    function go(n::Int64, a::BigInt, b::BigInt)::BigInt
        if n==0 
            return a
        else 
            return go(n-1,b,a+b)
        end
    end
    go(n,BigInt(0),BigInt(1))
end
```




    fib (generic function with 1 method)




```julia
@time fib(10000);
```

      0.044254 seconds (56.98 k allocations: 5.639 MiB)


### mutating functions
usually 
1. ends with `!`, for example `push!`, `pop!`
2. mutates first argument

### Structures
defined using `struct` keywords


```julia
struct Student  # capitalize by convention
    name::String  # can specify types when defining
    major::Symbol
    age::Int  
end
s1 = Student("Mike",:Statistics,17)
@show s1.name;  # can access its fields
```

    s1.name = "Mike"


# 4. Metaprogramming
macro: Expression => Expression


```julia
@time 1+1
```

      0.000002 seconds (4 allocations: 160 bytes)





    2




```julia
macro sayhi(name)
    return quote        # quote ... end creates an Expr
        "Hi, $($name)!"
    end
end

println(@sayhi "John")

println(typeof(quote println(1) end))   # can use quote expression end
println(typeof(:(println(1))))          # can also use :(expression)
```

    Hi, John!
    Expr
    Expr

