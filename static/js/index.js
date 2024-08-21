console.log("Hello!")

let startImage = document.querySelector("#startImage");
let endImage = document.querySelector("#endImage");
let startTimestamp = document.querySelector("#startTimestamp");
let endTimestamp = document.querySelector("#endTimestamp");
let startRange = document.querySelector("#startRange");
let endRange = document.querySelector("#endRange");

let fps = 24


window.addEventListener("load", async(e)=>{
    console.log("loaded!");

    let url = `/getVideoDuration`
    res = await fetch(url, {
        method: "GET"
    })
    let leftURL = await getImgURL("left", 0);
    let rightURL = await getImgURL("right", 0);

    startImage.src = leftURL;
    endImage.src = rightURL;

})

async function getImgURL(side, frameNum){
    let url = `/getContentByTime/${side}/?timestamp=${frameNum}`
    res = await fetch(url, {
        method: "GET"
    });
    let blob = await res.blob();
    console.log(blob);

    let imgURL = URL.createObjectURL(blob);
    return imgURL;
}

startRange.addEventListener("change", async (e)=>{
    startTimestamp.value = startRange.value/fps;
    let timestamp = Number(startRange.value)/fps;

    let leftURL = await getImgURL("left", timestamp);
    startImage.src = leftURL;
});

endRange.addEventListener("change", async (e) => {
  endTimestamp.value = endRange.value / fps;
  let timestamp = Number(endRange.value) / fps;

  let righttURL = await getImgURL("left", timestamp);
  endImage.src = righttURL;
});