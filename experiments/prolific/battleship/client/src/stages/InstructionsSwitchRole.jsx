import React from "react";
import { usePlayer, usePlayers, useRound } from "@empirica/core/player/classic/react";
import { Button } from "../components/Button";
import BoardComponent from "../components/BoardComponent.jsx";
import { SpotterBoardComponent } from "../components/SpotterBoardComponent.jsx";

export function InstructionsSwitchRole() {
  const player = usePlayer();
  const players = usePlayers();
  const round = useRound();

  return (
    <div style={{display:"flex", flexDirection:"column", alignItems:"center", gap:"10px"}}>
      <p style={{fontSize:"18px"}}>You will now play as the spotter. The spotter's role is to answer the captain's questions honestly and collaboratively.</p>
      <p style={{fontSize:"18px"}}>The spotter can see the entire board, but cannot fire at it. Below you can see the board how you saw it as the captain, and what it would have looked like to the spotter.</p>
      <div style={{display:"flex", flexDirection:"row", alignItems:"center", gap:"20px"}}>
        <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
          <u style={{fontSize: "1.3vw", marginLeft:"4vw"}}>Captain's View</u>
        <BoardComponent 
        init_tiles={player.stage.get("occTiles")}
        ships={player.stage.get("ships")}
        />
        </div>

        <div style={{display:"flex", flexDirection:"column", alignItems:"center"}}>
        <u style={{fontSize: "1.3vw", marginLeft:"4vw"}}>Spotter's View</u>
        <SpotterBoardComponent
              occ_tiles={player.stage.get("occTiles")}
              init_tiles={player.stage.get("trueTiles")}
              ships={player.stage.get("ships")}
          />
        </div>
      </div>
      <div style={{margin:"10px"}}>
      <Button handleClick={() => player.stage.set("introStage", 5)}>
        Continue
      </Button>
      </div>
    </div>
  );
}