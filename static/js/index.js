console.log("Hello!")

let startImage = document.querySelector("#startImage");
let endImage = document.querySelector("#endImage");
let startTimestamp = document.querySelector("#startTimestamp");
let endTimestamp = document.querySelector("#endTimestamp");
let startRange = document.querySelector("#startRange");
let endRange = document.querySelector("#endRange");



window.addEventListener("load", async(e)=>{
    console.log("loaded!");
})

startRange.addEventListener("change", (e)=>{
    startTimestamp.value = startRange.value;
})