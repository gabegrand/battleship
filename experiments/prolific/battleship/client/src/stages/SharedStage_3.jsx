import React, { useState } from "react";
import { usePlayer, useStage, useGame, useRound, useStageTimer } from "@empirica/core/player/classic/react";
import BoardComponent from "../components/BoardComponent.jsx";
import ShipsRemainingComponent from "../components/ShipsRemainingComponent.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";
import { noQuestion, timeoutAnswer } from "../../../utils/systemText.js";
import { HistoryComponent } from "../components/HistoryComponent.jsx";

export function SharedStage_3() {

    const round = useRound();
    const player = usePlayer();

    function getSummaryText() {
        if (round.get("question") == noQuestion) {
            return "You opted not to ask a question.";
        } else {
            if (round.get("answer") == timeoutAnswer) {
                return (<div style={{fontSize:"1vw"}}>
                    You asked <b>"{round.get('question')}"</b>. <i>The spotter did not answer.</i>
                </div>);
            }
            return (<div style={{fontSize:"1vw"}}>
                You asked <b>"{round.get('question')}"</b>. The spotter answered <b>"{round.get('answer')}"</b>.
            </div>);
        }
    }

    if (round.get("answer") == undefined) {
        round.set("answer", timeoutAnswer);
    }

    switch (player.round.get("role")) {
        case "spotter":
          player.stage.set("submit", true)
          return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}>
            
            <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
            <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                <ShipsRemainingComponent 
                    shipsStatus={round.get("shipsSunk")}
                />
                </div>
                <SpotterBoardComponent 
                    occ_tiles={round.get("occTiles")}
                    init_tiles={round.get("trueTiles")}
                    ships={round.get("ships")}
                />
                <HistoryComponent
                    grid = {round.get("occTiles")[0].length}/>
            </div>
            <div style={{margin: "20px", fontSize:"1vw"}}>
                <i>{round.get("question") == noQuestion ? "The captain opted not to ask a question. They are now selecting a tile to shoot at..." : "Thank you for answering. The captain is selecting a tile to shoot at next..."}</i>
            </div>
            </div>);
        case "captain":
          return (<div style={{display: "flex", flexDirection: "column", alignItems: "center"}}> 
                <div style={{display: "flex", flexDirection: "row", alignItems: "center"}}>
                <div style={{display: "flex", flexDirection: "column", margin: "30px", alignItems: "center", paddingTop:"50px"}}>
                <ShipsRemainingComponent 
                    shipsStatus={round.get("shipsSunk")}
                />
                </div>
                <BoardComponent 
                    init_tiles={round.get("occTiles")}
                    ships={round.get("ships")}
                />
                <HistoryComponent
                    grid = {round.get("occTiles")[0].length}/>
            </div>
            <p style={{margin: "20px"}} id="summary">
                {getSummaryText()}
            </p>
            <p style={{fontSize:"1vw"}}>Please click a tile to fire at it!</p>
            </div>);
        default:
          return <div>This is the shared stage. You have no role: something's gone wrong!</div>;
      }
}